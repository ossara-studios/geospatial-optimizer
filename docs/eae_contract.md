# EAE interface contract

This is the shared contract between the Python planner (IT side) and the EcoStruxure Automation Expert project (OT side).
**The EAE project must implement exactly these semantics.** The machine-readable node map is
[`config/opcua_nodes.yaml`](../config/opcua_nodes.yaml), and the limit values are
[`config/limits.yaml`](../config/limits.yaml).

The Python emulator (`geoopt emulator`) implements this contract. Use it as the reference
behaviour: if EAE and the emulator disagree, the emulator plus this document win.

## Topology (Plan A)
```
planner (OPC UA client) ──writes Rec.*──►  EAE OPC UA server  ◄──writes Rig.*── rig-sim (OPC UA client)
                        ◄─reads Chk/Appr/Done─                  ──reads Disp.*──►
```
EAE hosts the server and exposes the variables below. Both Python processes are clients, so EAE
needs no outbound OPC UA client blocks.

## Security
Local development uses `SecurityPolicy=None` and anonymous login. In production this link would
cross the IT/OT firewall with signed and encrypted OPC UA (Basic256Sha256) and certificate trust
lists.

## Variables (37 scalars; no strings or arrays)
| Group | Variable | Type | Writer | Meaning |
|---|---|---|---|---|
| Recommendation | `Rec.ReqId` | UInt32 | Planner | **Trigger.** Increments per recommendation and is written **last, as its own write call**. |
| | `Rec.HoleIdx` | UInt32 | Planner | Candidate hole index (Python maps it to a hole id) |
| | `Rec.CollarX`, `Rec.CollarY`, `Rec.CollarZ` | Double | Planner | Collar, local metres (origin in `config/default.yaml` → `data.origin`) |
| | `Rec.Azimuth`, `Rec.Dip` | Double | Planner | Degrees. Azimuth is clockwise from grid north; dip is **negative = down** |
| | `Rec.Depth` | Double | Planner | Planned depth, m |
| | `Rec.ProbOre`, `Rec.ExpOreM`, `Rec.Score` | Double | Planner | Shown on the HMI: mean P(Cu > cutoff), expected ore metres, score |
| Check | `Chk.ReqId` | UInt32 | EAE | Echo of the checked request (**written last**) |
| | `Chk.Status` | Int16 | EAE | 1 ACCEPTED, 2 CLIPPED, 3 REJECTED |
| | `Chk.Reason` | Int16 | EAE | 0 none, 1 out of bounds, 2 in no-go zone, 3 depth clipped, 4 dip out of range, 5 azimuth out of range |
| | `Chk.Depth` | Double | EAE | Depth after clipping, m |
| Approval | `Appr.ReqId` | UInt32 | EAE | Echo (**written last**) |
| | `Appr.Decision` | Int16 | EAE (operator on HMI) | 1 APPROVED, 2 DENIED |
| Dispatch | `Disp.ReqId` | UInt32 | EAE | Set only after APPROVED (**written last**). It's the rig's trigger. |
| | `Disp.HoleIdx` | UInt32 | EAE | Copied from `Rec.HoleIdx` |
| | `Disp.Depth` | Double | EAE | `Chk.Depth` (the clipped depth) |
| Rig | `Rig.State` | Int16 | Rig sim | 0 idle, 1 drilling, 2 complete, 9 fault |
| | `Rig.ReqId` | UInt32 | Rig sim | The request being drilled |
| | `Rig.DepthDrilled` | Double | Rig sim | Progress, m |
| | `Rig.Torque`, `Rig.WaterLpm`, `Rig.PowerKw` | Double | Rig sim | **Synthetic** telemetry, for display only |
| Done | `Done.ReqId` | UInt32 | EAE | Rig completion forwarded to the planner side (**written last**) |
| | `Done.MetresDrilled` | Double | EAE | Copied from `Rig.DepthDrilled` at completion |
| KPIs (display only) | `Kpi.Holes` | UInt32 | Planner | Holes drilled this run |
| | `Kpi.Metres`, `Kpi.OreMetres` | Double | Planner | Metres drilled; assayed metres above the cutoff |
| | `Kpi.NewOreMt`, `Kpi.NewCuKt` | Double | Planner | New ore found (Mt) and its contained Cu (kt): blocks first reached by an assayed ore sample, at an assumed 2.8 t/m³ |
| | `Kpi.DrillCostAudM` | Double | Planner | Metres × A$373.70/m (assumed all-in cost), A$ million |
| | `Kpi.BaselineMetres`, `Kpi.MetresSaved`, `Kpi.CostSavedAudM` | Double | Planner | Metres the near-ore rule needed to find as much new ore, the difference, and its cost (A$ M). `Kpi.BaselineMetres` = −1 means not available (savings then read 0); savings can be negative early on |

The planner writes the `Kpi.*` group after each hole's assays are in. EAE only displays it:
no EAE logic depends on it, and its values are INDICATIVE (see `geoopt.kpi`).

## ReqId rules
- `ReqId` only ever increases. On connect, the planner starts above every ReqId already on the
  server (`Rec`, `Chk`, `Appr` and `Done`), so a restart never reuses one.
- **Write order:** whoever writes a group writes the payload first and its `ReqId` (trigger or
  echo) **last, in a separate write**. OPC UA writes across several nodes aren't atomic, and EAE
  may scan between them. A reader acts on a group only once its `ReqId` matches.
- EAE acts once per new `Rec.ReqId` **higher** than the last one it handled. Lower or equal values
  are ignored and logged.
- Writes carry no source/server timestamps.

## EAE logic, per scan
1. **Forward completion.** If a request is dispatched and `Rig.State == 2` and
   `Rig.ReqId == Disp.ReqId`: write `Done.MetresDrilled = Rig.DepthDrilled`, then
   `Done.ReqId = Disp.ReqId`. If `Rig.State == 9`, raise an alarm on the HMI.
2. **New recommendation.** If `Rec.ReqId` > the last one handled, run the limit checks below and
   write `Chk.Status/Reason/Depth`, then `Chk.ReqId`.
   - REJECTED: stop. The planner excludes the hole and sends a new ReqId.
   - ACCEPTED or CLIPPED: show the recommendation on the HMI (hole, collar, depth after clip,
     ProbOre, ExpOreM, Score) and wait for the operator. Write `Appr.Decision`, then `Appr.ReqId`.
   - APPROVED: write `Disp.HoleIdx`, `Disp.Depth = Chk.Depth`, then `Disp.ReqId`.
   - DENIED: stop. The planner excludes the hole and sends a new ReqId.

## State machine (per ReqId)
```
RECOMMENDED → CHECKED ─┬─ REJECTED → planner excludes HoleIdx, issues new ReqId
                       └─ ACCEPTED/CLIPPED → AWAITING_APPROVAL ─┬─ DENIED → planner excludes, new ReqId
                                                                └─ APPROVED → DISPATCHED → DRILLING → COMPLETE → DONE
DONE → lab results for HoleIdx (above Chk.Depth) are already in the lab drop folder → planner ingests → next ReqId
```

## Limit semantics (EAE limit-check FB, in this order)
1. **Site bounds:** axis-aligned box on collar X/Y. Outside → REJECT, reason 1.
2. **No-go zones:** axis-aligned rectangles `x_min ≤ X ≤ x_max and y_min ≤ Y ≤ y_max`, or circles
   `hypot(X − cx, Y − cy) ≤ r`. Inside any → REJECT, reason 2. These zones are **illustrative**.
3. **Dip** outside `dip_range` (inclusive) → REJECT, reason 4.
4. **Azimuth** outside `azimuth_range` (inclusive) → REJECT, reason 5.
5. **Depth** over `max_depth_m` → CLIP to the maximum, status CLIPPED, reason 3. Otherwise ACCEPTED, reason 0.

Rejections take precedence over clipping. Minimum spacing between holes is a soft constraint in
the planner, not an EAE check. In replay mode a collar can't be moved, because a moved hole
doesn't exist in the dataset, so location problems reject rather than adjust.

## Timeouts (planner side)
- `timeouts.check_s` for `Chk.ReqId`.
- `timeouts.approval_s` for `Appr.ReqId` (null = wait indefinitely, logging every `log_wait_every_s`).
- `timeouts.drill_s` for `Done.ReqId` (null = indefinitely).

## Running the live loop
Use three terminals (or EAE in place of the first):
```bash
.venv/bin/geoopt emulator          # EAE stand-in; prompts y/n for approval
```
```bash
.venv/bin/geoopt rig-sim           # rig + lab simulator (OPC UA client)
```
```bash
.venv/bin/geoopt run-loop --bridge opcua --max-holes 5
```

## Pointing at real EAE
1. In `config/opcua_nodes.yaml`, replace each `id:` with EAE's `node_id:` (e.g. `ns=2;s=...`).
   Alternatively, set `namespace_uri` to EAE's namespace if the identifiers match.
2. Set `opcua.endpoint` in `config/default.yaml`.
3. Spike: `python scripts/opcua_roundtrip.py --depth 650`. Approve on the HMI and expect
   `status=CLIPPED depth=500.0 decision=APPROVED`.
4. Full loop: `geoopt rig-sim` and `geoopt run-loop --bridge opcua`.
