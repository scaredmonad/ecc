  ;; Memory declaration, initial size = 1 page (64KiB).
  (memory $mem 1)
  (export "memory" (memory $mem))

  ;; Global index to track the current top of the heap.
  (global $heap_ptr (mut i32) (i32.const 0))

  ;; Allocate block and return offset.
  (func $malloc (param $size i32) (result i32)
    (local $curr_heap_ptr i32) ;; Get curr_heap_ptr
    (set_local $curr_heap_ptr (global.get $heap_ptr))

    ;; Increase the heap pointer to allocate space.
    (global.set $heap_ptr (i32.add (global.get $heap_ptr) (local.get $size)))

    ;; Return the start address of the allocated block.
    (local.get $curr_heap_ptr)
  )
  (export "malloc" (func $malloc))

  ;; store<i32>(i32 offset, i32 value);
  (func $store_i32 (param $offset i32) (param $value i32)
    (i32.store (local.get $offset) (local.get $value))
  )
  (export "store_i32" (func $store_i32))

  ;; load<i32>(i32 offset);
  (func $load_i32 (param $offset i32) (result i32)
    (i32.load (local.get $offset))
  )
  (export "load_i32" (func $load_i32))

  ;; store<i64>(i64 offset, i64 value);
  (func $store_i64 (param $offset i32) (param $value i64)
    (i64.store (local.get $offset) (local.get $value)))
  (export "store_i64" (func $store_i64))

  ;; load<i64>(i64 offset);
  (func $load_i64 (param $offset i32) (result i64)
    (i64.load (local.get $offset)))
  (export "load_i64" (func $load_i64))

  ;; store<f32>(f32 offset, f32 value);
  (func $store_f32 (param $offset i32) (param $value f32)
    (f32.store (local.get $offset) (local.get $value)))
  (export "store_f32" (func $store_f32))

  ;; load<f32>(f32 offset);
  (func $load_f32 (param $offset i32) (result f32)
    (f32.load (local.get $offset)))
  (export "load_f32" (func $load_f32))

  ;; store<f64>(f64 offset, f64 value);
  (func $store_f64 (param $offset i32) (param $value f64)
    (f64.store (local.get $offset) (local.get $value)))
  (export "store_f64" (func $store_f64))

  ;; load<f64>(f64 offset);
  (func $load_f64 (param $offset i32) (result f64)
    (f64.load (local.get $offset)))
  (export "load_f64" (func $load_f64))
