# Troubleshooting

## Uploads fail with 413 Payload Too Large
The file exceeds `NIMBUS_MAX_UPLOAD_MB`. Either raise the limit in the Helm values or ask the
user to split the file. Remember the ingress controller has its own `proxy-body-size` limit,
which must be at least as large as the gateway limit.

## Clients do not receive sync notifications
1. Check that sync workers are running: `kubectl get pods -l app=sync-worker -n nimbus`.
2. Check consumer lag on the `sync-events` topic. Lag above 10,000 messages means workers are
   overloaded; scale the deployment up.
3. Make sure the client WebSocket is not blocked by a corporate proxy (look for `1006` close codes).

## Error: token_expired loops
If a client keeps receiving `token_expired` right after refreshing, its clock is probably
skewed. JWT validation allows only 30 seconds of clock skew. Ask the user to enable automatic
time sync.

## High Postgres CPU
Usually caused by the `janitor` job scanning `versions` without an index. Verify that the
`versions_created_at_idx` index exists. Run `EXPLAIN ANALYZE` on the janitor query to confirm.

## Gateway out-of-memory kills
The gateway should stream uploads. OOM kills typically mean a client is sending a
non-multipart upload larger than 64 MiB through the legacy `/v0/upload` endpoint. Block the
legacy endpoint or raise the memory limit temporarily.
