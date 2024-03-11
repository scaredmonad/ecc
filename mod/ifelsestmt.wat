(module
  (func $signum (param $n i32) (result i32))
    (local $sign i32)
    (set_local $sign (i32.const 0))
  (if (i32.gt_s (get_local $n) (i32.const 0))
    (then
      (set_local $sign (i32.const 1))
    (else
      (set_local $sign (i32.sub (i32.const 0) (i32.const 1)))
      )
)
)