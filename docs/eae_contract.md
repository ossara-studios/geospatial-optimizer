# EAE interface contract

This is the shared contract between the Python planner (IT side) and the EcoStruxure Automation Expert project (OT side).
The EAE project must implement exactly these semantics. The machine-readable node map is in
[`config/opcua_nodes.yaml`](../config/opcua_nodes.yaml), and the limit values are in [`config/limits.yaml`](../config/limits.yaml).

## Topology (Plan A)
EAE hosts the OPC UA server and exposes the variables below. The planner (and later the rig simulator) are OPC UA **clients**.
The Python emulator (`geoopt emulator`) exposes the same node map and logic, so the same client code runs against either one.

## Security
Local development uses `SecurityPolicy=None` and anonymous login. In production this link would cross the IT/OT firewall with signed and encrypted OPC UA (Basic256Sha256) and certificate trust lists.

## Variables (Phase 0 subset; all scalars, no strings or arrays)

| Variable | Type | Writer | Meaning |
|---|---|---|---|
| `Rec.ReqId` | UInt32 | Planner | Increments per recommendation. Writing it is the trigger. |
| `Rec.Depth` | Double | Planner | Planned depth, m |
| `Chk.ReqId` | UInt32 | EAE | Echo of the checked request |
| `Chk.Status` | Int16 | EAE | 1 ACCEPTED, 2 CLIPPED, 3 REJECTED |
| `Chk.Reason` | Int16 | EAE | 0 none, 1 out of bounds, 2 no-go zone, 3 depth clipped, 4 dip out of range, 5 azimuth out of range |
| `Chk.Depth` | Double | EAE | Depth after clipping, m |
| `Appr.ReqId` | UInt32 | EAE | Echo of the approved/denied request |
| `Appr.Decision` | Int16 | EAE (operator via HMI) | 1 APPROVED, 2 DENIED |

The full map (collar, azimuth/dip, scores, dispatch, rig, done) is added in Phase 7.

## ReqId rules
- `ReqId` only ever increases. On connect, the planner uses `max(Rec.ReqId, Chk.ReqId, Appr.ReqId) + 1`, so a restart never reuses an ID.
- **Write order:** the planner writes every `Rec.*` payload field first, then writes `Rec.ReqId` as a **separate, final** write call. OPC UA writes across several nodes are not atomic, and EAE may run between writes.
- EAE acts once per new `Rec.ReqId` that is **higher** than the last one it handled. A lower or equal value is ignored (and logged).
- EAE writes the result fields first and the echo (`Chk.ReqId`, `Appr.ReqId`) **last**. The planner reads the result fields only after the echo matches.
- Writes carry no source/server timestamps.

## State machine (per ReqId)
```
RECOMMENDED → CHECKED ─┬─ REJECTED → planner excludes the hole, issues a new ReqId
                       └─ ACCEPTED/CLIPPED → AWAITING_APPROVAL ─┬─ DENIED → planner excludes, new ReqId
                                                                └─ APPROVED → (Phase 7: DISPATCHED → DRILLING → COMPLETE)
```

## Limit semantics
- Depth over `max_depth_m` → CLIPPED to the maximum, reason 3.
- Phase 7 adds: site bounding box and no-go rectangles/circles on the collar (REJECT, reasons 1/2); dip/azimuth range (REJECT, reasons 4/5). No-go zones are **illustrative**.
- Minimum spacing between holes is a planner-side soft constraint, not an EAE check.

## Timeouts
The planner waits `timeouts.check_s` for `Chk.ReqId`. Approval waits `timeouts.approval_s` (null = indefinitely, logging every `log_wait_every_s`).

## Pointing at real EAE
1. In `config/opcua_nodes.yaml`, replace each `id:` with the EAE `node_id:` (e.g. `ns=2;s=...`), or set `namespace_uri` to EAE's namespace if the identifiers match.
2. Run `python scripts/opcua_roundtrip.py --endpoint opc.tcp://<eae-host>:<port> --depth 650`, approve on the HMI, and expect `status=CLIPPED depth=500.0 decision=APPROVED`.
