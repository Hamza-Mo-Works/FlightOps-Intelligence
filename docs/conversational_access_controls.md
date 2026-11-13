# Conversational access controls

The app backend validates a structured analysis plan before building SQL.
Allowed domains are OPS, AIRPORT and CARRIER. Their metric IDs are respectively
M001–M004, M005–M008 and M009–M010. Gold table names are selected in Python,
not accepted as user-provided SQL identifiers.

The SQL backend validates numerator/denominator identifiers from semantic
metadata. Supported operations are metric calculation, ranking, entity comparison,
daily/monthly trends and a two-period comparison. Date bounds are limited to 2025.
Airline/airport values are checked and supplied as SQL parameters.

OPS does not accept airline or airport filters. AIRPORT does not accept airline
filters; CARRIER does not accept airport filters. A selected domain scopes the
analytical interface; it is not a user authorization or row-level security system.

Warehouse access uses the configured app identity. Unity Catalog permissions,
warehouse entitlements and authentication must enforce actual access. Prompt
instructions and the UI domain selector do not substitute for those controls.

Local tests cover date bounds, unsupported filters, cross-domain metrics,
identifier rejection and parameter binding. They do not prove the deployed
principal has the intended permissions.
