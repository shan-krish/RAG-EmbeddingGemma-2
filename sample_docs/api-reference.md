# API Reference

Base URL: `https://api.nimbus.example`. All endpoints require `Authorization: Bearer <token>`.

## Files

### `GET /v1/files?path=/folder`
Lists files in a folder. Supports `cursor` pagination; page size is 100.

### `POST /v1/uploads`
Request body: `{"path": "/docs/report.pdf", "size": 1048576, "sha256": "..."}`.
Returns `{"upload_url": "...", "upload_id": "..."}`. Fails with `409` if a file with the same
path and checksum already exists, and with `507` if the user's quota is exhausted.

### `DELETE /v1/files/{id}`
Moves a file to trash. Trash is emptied after 30 days.

## Shares

### `POST /v1/shares`
Body: `{"path": "/team", "email": "a@b.c", "role": "editor"}`. Sends an invite email.

### `DELETE /v1/shares/{id}`
Revokes a share immediately; connected clients of the grantee are notified.

## Rate limits
Each user may make 600 requests per minute. Exceeding it returns `429` with a `Retry-After`
header. Upload URL creation is limited separately to 60 per minute.

## Errors
Errors use the shape `{"error": "code", "message": "human readable"}`. Common codes:
`token_expired`, `forbidden`, `not_found`, `quota_exceeded`, `rate_limited`.
