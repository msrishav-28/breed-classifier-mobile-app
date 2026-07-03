# Privacy facts

This document states what the app actually does with data, as implemented.
Use it as the source of truth for a hosted privacy policy and the Play
Console data-safety form.

## Summary

The app collects nothing, transmits nothing, and has no network access.

## Details

- **No network.** The app does not declare the `INTERNET` permission; it is
  technically incapable of transmitting data.
- **Permissions.** Exactly one: `CAMERA`, used only while the capture screen
  is open. Gallery photos come through the system photo picker, which
  requires no storage permission and exposes only the images the user picks.
- **Photos.** Stored in app-internal storage (`filesDir/images`), invisible
  to other apps, excluded from OS backups, deleted when the user deletes the
  history entry, clears history, or uninstalls.
- **Classification history.** Stored in a local Room database, excluded from
  OS backups, user-deletable in-app (per entry and clear-all).
- **Reports.** PDF reports are generated on demand into app-internal storage
  and leave the device only when the user explicitly shares one via the
  system share sheet.
- **On-device inference.** The TFLite model runs locally; images are never
  uploaded anywhere.
- **No analytics, no crash reporting, no identifiers.** The app contains no
  third-party SDKs that collect data and never reads device identifiers.

If any future change adds networking, analytics, or cloud sync, this
document and the public privacy policy must be updated first.
