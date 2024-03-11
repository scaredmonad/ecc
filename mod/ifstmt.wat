(module
  (func $main )
    (if (i32.gt_s (get_local $a) (i32.const 5))
      (then
        (set_local $a (i32.const 5))
        )
  )
)