use crate::parser::*;
use crate::scope::*;
use crate::visitor::*;
use std::cell::RefCell;
use std::rc::Rc;

// Level of indirection that allows any node to change the current scope. Why?
//  1. Contextual typing (terms of a context are well-defined)
//  2. Lookups and patching (bottom-up search of term(s) and beta reduction)
//  3. Expression-level typing (1 + 2.3 is ambiguous, implicit casts are illegal)
pub(crate) struct SemanticPass {
    pub(crate) current_scope: Option<Rc<RefCell<Scope>>>,
}

impl Default for SemanticPass {
    fn default() -> Self {
        Self {
            current_scope: None,
        }
    }
}

impl ProgramVisitor for SemanticPass {
    fn visit_program(&mut self, program: &mut Program) {
        let program_scope = Scope::new();

        program.scope = Some(Rc::clone(&program_scope));
        self.current_scope = Some(program_scope);

        for declaration in &mut program.declarations {
            self.visit_declaration(declaration);
        }
    }

    fn visit_function_declaration(&mut self, func_decl: &mut FunctionDeclaration) {
        if let Some(current_scope) = &self.current_scope {
            let func_scope = Scope::add_child(
                current_scope,
                Some(ASTNode::FunctionDeclaration(func_decl.clone())),
            );

            func_decl.scope = Some(Rc::clone(&func_scope));
            self.current_scope = Some(func_scope);

            for statement in &mut func_decl.body {
                statement.accept(self);
            }
        }
    }

    fn visit_variable_declaration(&mut self, func_decl: &mut VariableDeclaration) {
        if let Some(current_scope) = &self.current_scope {
            let func_scope = Scope::add_child(
                current_scope,
                Some(ASTNode::VariableDeclaration(func_decl.clone())),
            );

            func_decl.scope = Some(Rc::clone(&func_scope));
            self.current_scope = Some(func_scope);
        }
    }
}
