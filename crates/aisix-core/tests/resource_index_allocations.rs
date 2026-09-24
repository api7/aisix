use aisix_core::resource::{Resource, ResourceEntry};
use aisix_core::snapshot::ResourceTable;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;
thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = f();
    let count = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    (result, count)
}

#[derive(Debug)]
struct Item(String);
impl Resource for Item {
    fn id(&self) -> &str {
        ""
    }
    fn name(&self) -> &str {
        &self.0
    }
    fn kind() -> &'static str {
        "items"
    }
}

#[test]
fn copied_indices_share_storage_but_keep_mutations_isolated() {
    let source = ResourceTable::new();
    for i in 0..4096 {
        source.insert(ResourceEntry::new(
            format!("resource-{i}"),
            Item(format!("name-{i}")),
            1,
        ));
    }
    let generation = source.generation();
    let (copy, allocated) = allocations(|| source.clone());
    assert_eq!(copy.generation(), generation);
    copy.insert(ResourceEntry::new("resource-0", Item("renamed".into()), 2));
    copy.remove("resource-1");
    assert_ne!(copy.generation(), generation);
    assert_eq!(source.generation(), generation);
    assert_eq!(source.len(), 4096);
    assert_eq!(copy.len(), 4095);
    assert_eq!(source.get_by_name("name-0").unwrap().revision, 1);
    assert!(source.get_by_name("name-1").is_some());
    assert!(source.get_by_name("renamed").is_none());
    assert!(copy.get_by_name("name-0").is_none());
    assert!(copy.get_by_name("name-1").is_none());
    assert_eq!(copy.get_by_name("renamed").unwrap().revision, 2);
    drop(source);
    assert!(copy.get_by_id("resource-4095").is_some());
    // Shard storage still allocates, but duplicating each id/name does not.
    assert!(
        allocated < 4096,
        "copy allocated {allocated} times for 4096 rows"
    );
}

#[test]
fn name_lookup_does_not_allocate_an_id_string() {
    let table = ResourceTable::new();
    table.insert(ResourceEntry::new("resource", Item("name".into()), 1));
    let (entry, allocated) = allocations(|| table.get_by_name("name"));
    assert_eq!(entry.unwrap().id, "resource");
    assert_eq!(
        allocated, 0,
        "name lookup must only acquire existing handles"
    );
}

#[test]
fn filtered_entries_do_not_collect_or_clone_unrelated_rows() {
    let table = ResourceTable::new();
    for i in 0..4096 {
        table.insert(ResourceEntry::new(
            format!("key-{i}"),
            Item(format!("name-{i}")),
            1,
        ));
    }
    let held = table.get_by_id("key-0").unwrap();
    let none = table.matching_entries(|entry| {
        if entry.id == held.id {
            assert_eq!(std::sync::Arc::strong_count(&held), 2);
        }
        false
    });
    assert!(none.is_empty());
    let selected = table.matching_entries(|entry| entry.id == "key-0");
    assert_eq!(selected.len(), 1);
    assert!(std::sync::Arc::ptr_eq(&selected[0], &held));
}
