package com.livestock.recognition

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

/**
 * Main entry point for the Cattle and Buffalo Recognition application.
 * This activity serves as the launcher and navigation hub.
 */
class MainActivity : AppCompatActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }
}