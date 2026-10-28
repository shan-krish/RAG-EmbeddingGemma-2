# Authentication and Authorization

## Overview
Nimbus uses OAuth 2.0 with short-lived JWT access tokens and long-lived refresh tokens.
Access tokens expire after **15 minutes**; refresh tokens expire after **30 days** of inactivity.

## Obtaining tokens

Use the authorization-code flow with PKCE for all first-party clients:

```bash
curl -X POST https://auth.nimbus.example/oauth/token \
  -d grant_type=authorization_code \
  -d code=$CODE -d code_verifier=$VERIFIER -d client_id=nimbus-desktop
```

The response contains `access_token`, `refresh_token` and `expires_in`.

## Refreshing tokens
When the gateway returns `401` with `error=token_expired`, call the token endpoint with
`grant_type=refresh_token`. Refresh tokens are rotated: every refresh returns a new refresh
token and invalidates the old one. Reusing an old refresh token revokes the whole session
as a theft-detection measure.

## Service-to-service auth
Internal services authenticate with mTLS. Certificates are issued by the internal CA and
rotated every 24 hours by the `cert-rotator` sidecar. Never use user tokens between services.

## Permissions
Shares grant `viewer`, `editor` or `owner` roles on a folder. Roles are inherited by
sub-folders. Permission checks happen in the gateway via the `authz.Check(user, path, action)`
function, which caches decisions for 30 seconds.

## Resetting a password
Users reset their password from the login page using the "Forgot password" link. A reset
email with a single-use link (valid for 1 hour) is sent. Resetting a password revokes all
existing refresh tokens for that user.
