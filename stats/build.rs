fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=/Library/Frameworks/Python.framework/Versions/3.13/lib");
    
    // Link against the Python framework on macOS
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=python3.13");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Python");
        println!("cargo:rustc-link-arg=-F/Library/Frameworks");
    }

    // Force rebuild if the build script changes
    println!("cargo:rerun-if-changed=build.rs");
}

