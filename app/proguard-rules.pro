# TensorFlow Lite uses JNI; keep its runtime classes and native bindings.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# The GPU delegate is not bundled; silence references from tensorflow-lite core.
-dontwarn org.tensorflow.lite.gpu.**
