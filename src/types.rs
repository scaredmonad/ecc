use crate::parser::Type;
use std::collections::HashSet;
use std::hash::Hash;
use std::ops::{Add, Sub};

// 1. Ensure elements are unique with strict reflexivity (`Eq`) including crate::parser::Type.
// 2. Lookups without inadvertently duplicating entries or losing track of substitutions.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Set<T: Eq + Hash + PartialEq> {
    elements: HashSet<T>,
}

impl<T: Eq + Hash + Clone> Set<T> {
    fn new() -> Self {
        Set {
            elements: HashSet::new(),
        }
    }
}

impl<T: Eq + Hash + Clone> Add for Set<T> {
    type Output = Set<T>;

    fn add(self, other: Self) -> Self::Output {
        let mut new_set = self.elements.clone();

        for elem in other.elements.iter() {
            new_set.insert(elem.clone());
        }

        Set { elements: new_set }
    }
}

impl<T: Eq + Hash + Clone> Sub for Set<T> {
    type Output = Set<T>;

    fn sub(self, other: Self) -> Self::Output {
        let mut new_set = self.elements.clone();

        for elem in other.elements.iter() {
            new_set.remove(elem);
        }

        Set { elements: new_set }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_set_add() {
        let mut set1 = Set::new();
        set1.elements.insert(1);
        let mut set2 = Set::new();
        set2.elements.insert(2);
        let result_set = set1 + set2;

        let mut expected_set = Set::new();
        expected_set.elements.insert(1);
        expected_set.elements.insert(2);

        assert_eq!(result_set, expected_set);
        println!("assert_set_add passed.");
    }

    fn assert_set_sub() {
        let mut set1 = Set::new();
        set1.elements.insert(1);
        set1.elements.insert(2);
        let mut set2 = Set::new();
        set2.elements.insert(2);
        let result_set = set1 - set2;

        let mut expected_set = Set::new();
        expected_set.elements.insert(1);

        assert_eq!(result_set, expected_set);
        println!("assert_set_sub passed.");
    }

    #[test]
    fn test_set_operations() {
        assert_set_add();
        assert_set_sub();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Term {
    Var(String),
    Abs(String, Box<Term>),
    App(Box<Term>, Box<Term>),
    MonoType(Box<Term>),
    PolyType(Vec<Type>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TermOrContext {
    Term(Box<Term>),
    Context(Box<TypeContext>),
}

impl Term {
    // Matches a Term against a specific type variable. Accordingly, it doesn't work
    // for nested structures or polymorphic types (cf. other implementations).
    fn contains_var(&self, var_term: &Term) -> bool {
        match (self, var_term) {
            (Term::Var(name), Term::Var(var_name)) => name == var_name,
            (Term::Abs(_, body), _) => body.contains_var(var_term),
            (Term::App(f, arg), _) => f.contains_var(var_term) || arg.contains_var(var_term),
            (Term::MonoType(term), _) => term.contains_var(var_term),
            _ => false,
        }
    }
}

trait Free {
    fn free_vars(&self) -> Set<String>;
}

impl Free for Term {
    fn free_vars(&self) -> Set<String> {
        match self {
            Term::Var(name) => {
                let mut vars = Set::new();
                vars.elements.insert(name.clone());
                vars
            }

            Term::Abs(param, body) => {
                let mut vars = body.free_vars();
                vars.elements.remove(param);
                vars
            }

            Term::App(f, arg) => {
                let mut vars = f.free_vars();

                for var in arg.free_vars().elements.iter() {
                    vars.elements.insert(var.clone());
                }

                vars
            }

            Term::MonoType(_) | Term::PolyType(_) => Set::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Substitution {
    mappings: Set<(Term, Term)>,
}

impl Substitution {
    fn make(&mut self, from: Term, to: Term) {
        self.mappings.elements.insert((from, to));
    }

    pub fn apply(&self, toc: TermOrContext) -> TermOrContext {
        match toc {
            // Recursively apply substitution to the Term.
            TermOrContext::Term(t) => TermOrContext::Term(Box::new(self.apply_term(&t))),

            // Apply substitution to each Term in the TypeContext.
            TermOrContext::Context(ctx) => {
                let mut new_ctx = TypeContext::default();

                for term in ctx.definitions.elements.iter() {
                    new_ctx.definitions.elements.insert(self.apply_term(term));
                }

                TermOrContext::Context(Box::new(new_ctx))
            }
        }
    }

    fn apply_term(&self, term: &Term) -> Term {
        match term {
            Term::Var(name) => {
                for (var, typ) in &self.mappings.elements {
                    if let Term::Var(var_name) = var {
                        if var_name == name {
                            return typ.clone();
                        }
                    }
                }

                term.clone()
            }

            Term::Abs(param, body) => Term::Abs(param.clone(), Box::new(self.apply_term(body))),

            Term::App(f, arg) => {
                Term::App(Box::new(self.apply_term(f)), Box::new(self.apply_term(arg)))
            }

            // MonoType and PolyType do not contain variables that need substitution.
            _ => term.clone(),
        }
    }
}

impl core::default::Default for Substitution {
    fn default() -> Self {
        Self {
            mappings: Set::new(),
        }
    }
}

impl Add for Substitution {
    type Output = Self;

    // We combine by addition over subst.combine().
    fn add(self, other: Self) -> Self::Output {
        let mut combined = self.mappings.clone();

        for elem in other.mappings.elements {
            combined.elements.insert(elem);
        }

        Substitution { mappings: combined }
    }
}

#[test]
fn assert_empty_substitution() {
    let subst1 = Substitution::default();
    let subst2 = Substitution::default();
    assert_eq!(subst1, subst2);
}

#[test]
fn assert_make_substitutions() {
    let mut subst1 = Substitution::default();
    subst1.make(Term::Var("A".into()), Term::Var("B".into()));
    assert_eq!(true, subst1.mappings.elements.len() == 1);
}

#[test]
fn assert_can_combine_substitutions() {
    let mut subst1 = Substitution::default();
    subst1.make(Term::Var("A".into()), Term::Var("B".into()));
    let mut subst2 = Substitution::default();
    subst2.make(Term::Var("A".into()), Term::Var("B".into()));
    dbg!(subst1.clone());
    assert_eq!(subst1, subst2);
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TypeContext {
    definitions: Set<Term>,
}

impl core::default::Default for TypeContext {
    fn default() -> Self {
        Self {
            definitions: Set::new(),
        }
    }
}

impl TypeContext {
    pub fn generalize(&self, term: &Term) -> Term {
        let term_free_vars = term.free_vars();
        let mut context_free_vars = Set::new();

        for t in self.definitions.elements.iter() {
            if let Term::Var(name) = t {
                context_free_vars.elements.insert(name.clone());
            }
        }

        let mut generalizable_vars = Set::new();

        for var in term_free_vars.elements.iter() {
            if !context_free_vars.elements.contains(var) {
                generalizable_vars.elements.insert(var.clone());
            }
        }

        if generalizable_vars.elements.is_empty() {
            term.clone()
        } else {
            term.clone()
        }
    }
}

fn unify(t1: &Term, t2: &Term) -> Result<Substitution, String> {
    match (t1, t2) {
        // Γ ⊢ x:τ ≡ x:τ
        (Term::Var(name1), Term::Var(name2)) if name1 == name2 => Ok(Substitution::default()),

        // Γ ⊢ x:σ ≡ τ such that τ does not contain x
        (Term::Var(name), term) | (term, Term::Var(name))
            if !term.contains_var(&Term::Var(name.clone())) =>
        {
            let mut subst = Substitution::default();
            subst.make(Term::Var(name.clone()), (*term).clone());
            Ok(subst)
        }

        // Γ ⊢ λx.σ1 ≡ λx.σ2 => Γ, x:σ1 ⊢ σ1 ≡ σ2
        (Term::Abs(param1, body1), Term::Abs(param2, body2)) => {
            let body_subst = unify(&*body1, &*body2)?;
            let param_subst = unify(&Term::Var(param1.clone()), &Term::Var(param2.clone()))?;
            let combined_subst = param_subst + body_subst;
            Ok(combined_subst)
        }

        // Γ ⊢ (σ1 σ2) ≡ (τ1 τ2) => Γ ⊢ σ1 ≡ τ1 ∧ σ2 ≡ τ2
        (Term::App(f1, arg1), Term::App(f2, arg2)) => {
            let func_subst = unify(&**f1, &**f2)?;
            let applied_arg1 = func_subst.apply(TermOrContext::Term(Box::new(**arg1)));
            let applied_arg2 = func_subst.apply(TermOrContext::Term(Box::new(**arg2)));

            let arg_subst = match applied_arg1 {
                TermOrContext::Term(app_t_1) => {
                    match applied_arg2 {
                        TermOrContext::Term(app_t_2) => unify(&app_t_1, &app_t_2)?,
                        _ => Substitution::default(), /*??*/
                    }
                }
                _ => Substitution::default(), /*??*/
            };
            Ok(func_subst + arg_subst)
        }

        // Γ ⊢ σ ≡ τ for MonoTypes
        (Term::MonoType(inner1), Term::MonoType(inner2)) => unify(&**inner1, &**inner2),

        _ => Err("Cannot unify different types".into()),
    }
}
