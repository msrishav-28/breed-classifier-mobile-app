# Play Store Descriptions

## Short Description

Offline cattle and buffalo breed recognition for Gir, Hallikar, Murrah,
Sahiwal, and Tharparkar.

## Full Description

Breed Classifier helps identify supported Indian cattle and buffalo breeds
from a photo, entirely on your Android device.

Supported breeds in this release:

- Gir
- Hallikar
- Murrah
- Sahiwal
- Tharparkar

The app uses a bundled TensorFlow Lite model and does not need internet
connectivity. Take a photo with the camera or choose one through the Android
photo picker, then review the top breed matches, confidence level, image
quality warnings, and breed details from the bundled catalog.

Measured model performance for this release candidate:

- 88.00% top-1 accuracy on a held-out real-photo test set
- 100.00% top-3 accuracy on the same test set
- 63.33% minimum per-breed recall

Features:

- Offline on-device recognition
- Top-3 breed predictions with confidence
- Quality warnings for dark, blurry, low-resolution, or low-contrast images
- Breed details including species, origin, primary use, and characteristics
- Local classification history
- PDF report export through the Android share sheet
- No account, no ads, no analytics, no internet permission

Privacy:

Photos and history stay on the device. The app declares only the camera
permission, stores photos in app-internal storage, and transmits nothing.

Important limitations:

This release recognises only the five supported breeds listed above. Results
are decision support, not a veterinary, registry, or legal breed certification.
Use clear side-view animal photos where possible; poor lighting, occlusion,
young animals, mixed breeds, and breeds outside the supported set can reduce
accuracy.
