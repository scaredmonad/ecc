use crate::parser::*;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone)]
pub(crate) struct Scope {
    pub(crate) parent: Option<Weak<RefCell<Scope>>>,
    pub(crate) children: Vec<Rc<RefCell<Scope>>>,
    pub(crate) curr_node: Option<ASTNode>,
}

impl Scope {
    pub(crate) fn new() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            parent: None,
            children: Vec::new(),
            curr_node: None,
        }))
    }

    pub(crate) fn add_child(
        parent: &Rc<RefCell<Self>>,
        curr_node: Option<ASTNode>,
    ) -> Rc<RefCell<Self>> {
        let child = Rc::new(RefCell::new(Self {
            parent: Some(Rc::downgrade(parent)),
            children: Vec::new(),
            curr_node,
        }));
        parent.borrow_mut().children.push(Rc::clone(&child));
        child
    }
}
