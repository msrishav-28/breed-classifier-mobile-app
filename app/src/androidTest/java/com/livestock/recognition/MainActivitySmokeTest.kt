package com.livestock.recognition

import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.livestock.recognition.ui.main.MainActivity
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Minimal on-device smoke test: the home screen launches and its primary
 * actions are visible. Run on a connected device with
 * `./gradlew connectedDebugAndroidTest`; CI compiles it to keep it healthy.
 */
@RunWith(AndroidJUnit4::class)
class MainActivitySmokeTest {

    @Test
    fun homeScreenShowsPrimaryActions() {
        ActivityScenario.launch(MainActivity::class.java).use {
            onView(withId(R.id.takePhotoButton)).check(matches(isDisplayed()))
            onView(withId(R.id.pickImageButton)).check(matches(isDisplayed()))
            onView(withId(R.id.historyButton)).check(matches(isDisplayed()))
        }
    }
}
