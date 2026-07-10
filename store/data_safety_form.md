# Play Console Data Safety Answers

Source of truth: `docs/PRIVACY.md`.

## Data Collection

Does the app collect or share user data?

Answer: No.

Reason: The app has no `INTERNET` permission, no analytics SDKs, no crash
reporting SDKs, no account system, and no cloud sync. Photos, history, and
reports are stored locally in app-internal storage unless the user explicitly
shares a PDF through the Android share sheet.

## Data Types

Declare no collected data types.

The app processes photos locally for classification, but does not collect,
transmit, sell, share, or upload them.

## Security Practices

- Data encrypted in transit: Not applicable; no data is transmitted.
- Users can request deletion: Not applicable for server-side data because no
  server-side data exists. Users can delete local history entries or clear all
  history in the app.
- Independent security review: No, unless the operator completes one before
  submission.

## Permissions

Declared permission:

- `CAMERA`: used only to capture a photo when the user opens the camera flow.

No storage permissions are requested. Gallery selection uses the Android
system photo picker.

## Data Sharing

No data is shared by the app. If a user explicitly shares a generated PDF via
the system share sheet, that is a user-directed action outside automatic app
collection.
