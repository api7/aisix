//! Heap the compiled resource-schema validators keep for the process
//! lifetime.
//!
//! [`SCHEMAS`] and [`LENIENT_SCHEMAS`] are statics built on the first
//! resource load and never dropped, so whatever they hold is a fixed cost
//! of every gateway. A validator library that inlines a fresh copy of each
//! `$ref` target held over 100 MiB here; this pins the live size.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use aisix_core::models::schema::{LENIENT_SCHEMAS, SCHEMAS};

struct CountingAllocator;

static LIVE: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Both validator sets together. They measure a few MiB; the bound leaves
/// room for new resources without admitting a per-`$ref` copy again.
const MAX_LIVE_BYTES: isize = 32 * 1024 * 1024;

#[test]
fn compiled_validators_stay_small() {
    let before = LIVE.load(Ordering::Relaxed);
    let strict = &*SCHEMAS;
    let lenient = &*LENIENT_SCHEMAS;
    let held = LIVE.load(Ordering::Relaxed) - before;
    // Keep both alive until after the measurement.
    std::hint::black_box((strict, lenient));
    eprintln!(
        "compiled validators hold {:.1} MiB",
        held as f64 / 1048576.0
    );
    assert!(
        held <= MAX_LIVE_BYTES,
        "compiled validators hold {held} bytes, over the {MAX_LIVE_BYTES}-byte bound"
    );
}
