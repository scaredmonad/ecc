use crate::parser::*;
use crate::scope::*;
use crate::visitor::*;
use std::cell::RefCell;
use std::rc::Rc;

// We use a Vec<String> because we can later iterate and collect into a
// string interning structure. Or maybe it should be an AsRef<T>?
#[derive(Debug, Clone)]
pub(crate) struct Printer {
    indent_level: u8,
    lines: Vec<String>,
}

impl Printer {
    fn new() -> Self {
        Printer {
            lines: Vec::new(),
            indent_level: 0,
        }
    }

    fn def_mod(&mut self) {
        self.lines.push("(module".to_string());
    }

    fn end_mod(&mut self) {
        self.lines.push(")".to_string());
    }

    fn def_global_var(&mut self, name: &str, var_type: &str, value: &str) {
        self.lines.push(format!(
            "  (global ${} ({} {}) ({}))",
            name, var_type, "const", value
        ));
    }

    // This just DEFINES (!), no assignment until Expression::Uninit is checked.
    fn def_local_var(&mut self, name: &str, var_type: &str) {
        self.lines.push(format!(
            "{}(local ${} {})",
            "  ".repeat(self.indent_level.into()),
            name,
            var_type
        ));
    }

    #[allow(unused_assignments)]
    fn rec_write_expr(expr: &Expression, scope: Option<Rc<RefCell<Scope>>>) -> String {
        match expr {
            Expression::Uninit => String::new(),
            Expression::IntLiteral(value) => {
                // let curr_non_term = &scope.as_ref().unwrap().borrow().curr_node;
                let mut expected_type = String::new();

                match &scope {
                    Some(s) => {
                        let curr_non_term = &s.borrow().curr_node;
                        expected_type = match curr_non_term {
                            Some(ASTNode::VariableDeclaration(var_decl)) => {
                                var_decl.data_type.to_string()
                            }
                            _ => "i32".into(),
                        };
                    }
                    _ => expected_type = "i32".into(), // why?
                                                       // This issue arises in the if stmt only (so far)
                };

                // let expected_type = match curr_non_term {
                //     Some(ASTNode::VariableDeclaration(var_decl)) => var_decl.data_type.to_string(),
                //     _ => "i32".into(),
                // };

                format!("({}.const {})", expected_type, value)
            }
            Expression::BoolLiteral(value) => {
                let bool_val = if *value { "1" } else { "0" };
                format!("(i32.const {})", bool_val)
            }
            Expression::Variable(Identifier(name)) => format!("(get_local ${})", name),
            Expression::Binary(left, op, right) => {
                let lhs_str = Self::rec_write_expr(left, scope.clone());
                let rhs_str = Self::rec_write_expr(right, scope);
                let op_str = match op {
                    BinaryOp::Add => "i32.add",
                    BinaryOp::Sub => "i32.sub",
                    BinaryOp::Mul => "i32.mul",
                    BinaryOp::Div => "i32.div_s", /* signed 32-bit integer division, for unsigned use i32.div_u */
                };

                format!("({} {} {})", op_str, lhs_str, rhs_str)
            }
            // We only do signed comparisons!
            Expression::Comparison(left, op, right) => {
                let lhs_str = Self::rec_write_expr(left, scope.clone());
                let rhs_str = Self::rec_write_expr(right, scope);
                let op_str = match op {
                    CompareOp::StrictEqual => "i32.eq",
                    CompareOp::StrictUnequal => "i32.ne",
                    CompareOp::GreaterThan => "i32.gt_s",
                    CompareOp::LessThan => "i32.lt_s",
                    CompareOp::GreaterThanOrEqual => "i32.ge_s",
                    CompareOp::LessThanOrEqual => "i32.le_s",
                };

                format!("({} {} {})", op_str, lhs_str, rhs_str)
            }
            // Expression::Assignment(assignment_expr) => {},

            // This entire block doesn't work as expected bc:
            // - type params shouldn't be used at runtime
            // - malloc<T>(), sizeof<T>(), load<T>() and store<T> should work
            // in line with how we configure it in the prelude.
            // - so this is just a stub
            Expression::Call(call_expr) => {
                let callee = &call_expr.callee.0;
                let params = call_expr
                    .parameters
                    .iter()
                    .map(|e| {
                        let scope_clone = scope.clone();
                        Self::rec_write_expr(e, scope_clone)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{}\n(call ${})", params, callee)
            }
            _ => "".into(),
        }
    }

    fn assign(&mut self, var_name: &str, value: &str) {
        self.lines.push(format!(
            "{}(set_local ${} {})",
            "  ".repeat(self.indent_level.into()),
            var_name,
            value
        ));
    }

    fn def_func(&mut self, name: &str, params: Vec<(String, String)>, return_type: Option<String>) {
        let params_str = params
            .into_iter()
            .map(|(type_, id)| format!("(param ${} {})", id, type_))
            .collect::<Vec<_>>()
            .join(" ");
        let return_str =
            return_type.map_or(String::new(), |r_type| format!(" (result {})", r_type));
        self.lines.push(format!(
            "{}(func ${} {}{})",
            "  ".repeat(self.indent_level.into()),
            name,
            params_str,
            return_str
        ));
    }

    fn end_func(&mut self) {
        self.lines
            .push(format!("{})", "  ".repeat(self.indent_level.into())));
    }

    fn def_if(&mut self, condition: String) {
        self.lines.push(format!(
            "{}(if {}",
            "  ".repeat(self.indent_level.into()),
            condition
        ));
    }

    fn def_then(&mut self) {
        self.lines
            .push(format!("{}(then", "  ".repeat(self.indent_level.into()),));
    }

    fn def_else(&mut self) {
        self.lines
            .push(format!("{}(else", "  ".repeat(self.indent_level.into()),));
    }

    fn raw_append(&mut self, value: &str) {
        self.lines.push(value.into())
    }

    pub(crate) fn to_string(&self) -> String {
        self.lines.join("\n")
    }
}

pub(crate) struct WritePass {
    pub(crate) printer: Printer,
}

impl Default for WritePass {
    fn default() -> Self {
        Self {
            printer: Printer::new(),
        }
    }
}

impl ProgramVisitor for WritePass {
    fn visit_program(&mut self, program: &mut Program) {
        self.printer.def_mod();

        // self.printer.indent_level += 1; /* temp */
        // self.printer.raw_append(STD_LINEAR_PRELUDE);

        for declaration in &mut program.declarations {
            self.visit_declaration(declaration);
        }

        self.printer.end_mod();
    }

    fn visit_asm_block(&mut self, asm_block: &mut AsmBlock) {
        if let Expression::Assignment(ass) = &asm_block.target {
            if let Expression::Variable(ref ass_rhs) = *ass.right {
                let ass_rhs_ident = &ass_rhs.0;
                if ass_rhs_ident.as_str() != "default" {
                    core::panic!("Default printer expects `default` on all asm blocks.");
                }
            } else {
                core::panic!("Expected an ident RHS expression on asm block.");
            }
        }

        self.printer.raw_append(&asm_block.instr_field);
    }

    // We need to determine whether this is a global or local var.
    fn visit_variable_declaration(&mut self, var_decl: &mut VariableDeclaration) {
        self.printer
            .def_local_var(&var_decl.identifier.0, &var_decl.data_type.to_string());

        let expr_repr = Printer::rec_write_expr(&var_decl.value, var_decl.scope.clone());

        // dbg!(&var_decl.scope);

        // The stack model of WASM forces us to push all params, call and then
        //  do an empty set_local, which gets the return value for assignment.
        match &var_decl.value {
            Expression::Call(_) => {
                self.printer.raw_append(&expr_repr);
                self.printer.assign(&var_decl.identifier.0, "");
            }

            _ => {
                self.printer.assign(&var_decl.identifier.0, &expr_repr);
            }
        }
    }

    fn visit_statement(&mut self, statement: &mut Statement) {
        // self.printer.indent_level += 1;

        match statement {
            Statement::VariableDeclaration(var_decl) => self.visit_variable_declaration(var_decl),

            // Do not check for Expression, which doesn't hold AssignmentExpression in the parse phase.
            Statement::ExpressionStatement(expr) => {
                if let Expression::Assignment(ass) = expr {
                    self.printer
                        .assign(&ass.left.0, &Printer::rec_write_expr(&ass.right, None));
                }
            }

            Statement::Return(ret) => {
                self.printer.raw_append(&Printer::rec_write_expr(ret, None));
            }

            Statement::IfStatement(if_stmt) => {
                let condition = Printer::rec_write_expr(&if_stmt.condition, None);
                self.printer.def_if(condition);
                self.printer.indent_level += 1;
                self.printer.def_then();
                self.printer.indent_level += 1;

                for statement in &mut if_stmt.body {
                    statement.accept(self);
                }

                if let Some(alt) = &mut if_stmt.alternative {
                    self.printer.def_else();
                    self.printer.indent_level += 1;

                    for statement in alt
                    /*.iter_mut()*/
                    {
                        statement.accept(self);
                    }
                }

                self.printer.indent_level += 1;
                self.printer.end_func();
                self.printer.indent_level -= 1;
            }

            _ => {}
        }

        self.printer.indent_level -= 1; // /!\
    }

    fn visit_function_declaration(&mut self, func_decl: &mut FunctionDeclaration) {
        self.printer.indent_level += 1;
        let retkind = if func_decl.return_type == Type::Void {
            None
        } else {
            Some(func_decl.return_type.to_string())
        };
        self.printer.def_func(
            &func_decl.identifier.0,
            func_decl
                .parameters
                .iter()
                .map(|(t, id)| (t.to_string(), id.0.clone()))
                .collect::<Vec<(String, String)>>(),
            retkind,
        );
        self.printer.indent_level += 1;
        for statement in &mut func_decl.body {
            statement.accept(self);
        }
        self.printer.indent_level -= 1;
        self.printer.end_func();
        self.printer.indent_level = 0;
    }
}
