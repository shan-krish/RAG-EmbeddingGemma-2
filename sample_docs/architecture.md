# Nimbus Architecture

Nimbus is a small file-sync service. Clients upload files to the **gateway**, which stores
blobs in object storage and records metadata in Postgres.

## Components

### Gateway
The gateway is a Go HTTP service listening on port `8080`. It terminates TLS, validates
access tokens, and streams uploads directly to object storage without buffering whole files
in memory. Uploads larger than 64 MiB are split into 8 MiB parts (multipart upload).

### Metadata store
Postgres 16 holds the `files`, `versions` and `shares` tables. Every upload creates a new row
in `versions`; old versions are kept for 30 days and then removed by the `janitor` job.

### Sync workers
Workers consume the `sync-events` Kafka topic and push change notifications to connected
clients over WebSockets. A worker is stateless; scale horizontally by adding replicas.

## Data flow

1. Client requests an upload URL from `POST /v1/uploads`.
2. Gateway checks quota, returns a pre-signed URL valid for 15 minutes.
3. Client uploads the blob; object storage calls back the gateway webhook.
4. Gateway writes a `versions` row and publishes a `file.updated` event.
5. Sync workers fan the event out to every device of that user.

## Design decisions

We chose pre-signed URLs so that file bytes never pass through our own servers in the steady
state. This keeps the gateway cheap and lets object storage handle bandwidth spikes.
Postgres was preferred over a document database because shares and permissions are
relational and need transactional updates.
