plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "uk.local.ratwatch.watch"
    compileSdk = 35

    defaultConfig {
        applicationId = "uk.local.ratwatch.watch"
        minSdk = 30
        targetSdk = 35
        versionCode = 1
        versionName = "2.0.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("com.google.android.gms:play-services-wearable:18.2.0")
    implementation("androidx.wear:wear:1.3.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
