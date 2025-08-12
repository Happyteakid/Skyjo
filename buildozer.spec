[app]
title = Skyjo
package.name = skyjo
package.domain = org.example
source.dir = .
source.include_exts = py,kv,json
version = 0.1
# Kivy is required. If you add other libs later, put them here.
requirements = python3,kivy
orientation = portrait
fullscreen = 0

[buildozer]
log_level = 2
warn_on_root = 0

[android]
# Modern Android API (adjust if needed)
android.api = 34
# Let p4a pick appropriate NDK; uncomment to pin:
# android.ndk = 25b
archs = arm64-v8a,armeabi-v7a
