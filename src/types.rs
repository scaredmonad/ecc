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
    MonoType(Type),
    PolyType(Vec<Type>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Substitution {
    mappings: Set<(Term, Term)>,
}

impl core::default::Default for Substitution {
    fn default() -> Self {
        Self {
            mappings: Set::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TypeContext {
    definitions: Set<Term>,
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
