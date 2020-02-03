use std::cell::{Cell, Ref, RefCell, RefMut};
use std::cmp;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::rc::Rc;

use crate::operators as op;
use crate::{Device, Operator, Parameter, Shape, Tensor};

struct NodeData<'dev> {
    value: RefCell<Tensor<'dev>>,
    gradient: RefCell<Tensor<'dev>>,
}

struct DataRef<'arg, 'dev>
where
    'dev: 'arg,
{
    op: Rc<OperatorInfo<'arg, 'dev>>,
    vid: usize,
}

struct OperatorInfo<'arg, 'dev> {
    operator: Box<dyn Operator<'arg, 'dev> + 'arg>,
    args: Vec<DataRef<'arg, 'dev>>,
    rets: Vec<NodeData<'arg>>,
    forwarded: Cell<bool>,
    depth: usize,
}

struct OperatorInfoCmp<'arg, 'dev>(Rc<OperatorInfo<'arg, 'dev>>);

impl<'arg, 'dev> PartialEq for OperatorInfoCmp<'arg, 'dev> {
    fn eq(&self, other: &Self) -> bool {
        self.0.depth == other.0.depth
    }
}

impl<'arg, 'dev> Eq for OperatorInfoCmp<'arg, 'dev> {}

impl<'arg, 'dev> Ord for OperatorInfoCmp<'arg, 'dev> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.depth.cmp(&other.0.depth)
    }
}

impl<'arg, 'dev> PartialOrd for OperatorInfoCmp<'arg, 'dev> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.depth.cmp(&other.0.depth))
    }
}

pub struct Node<'arg, 'dev> {
    data: DataRef<'arg, 'dev>,
}

impl<'arg, 'dev> OperatorInfo<'arg, 'dev> {
    fn new<T: Operator<'arg, 'dev> + 'arg>(
        op: T,
        xs: &[&Node<'arg, 'dev>],
    ) -> OperatorInfo<'arg, 'dev> {
        let arg_shapes = xs
            .iter()
            .map(|x| x.inner_value().shape)
            .collect::<Vec<Shape>>();
        let ret_shapes = op.forward_shape(&arg_shapes);
        let args = xs
            .iter()
            .map(|x| DataRef {
                op: Rc::clone(&x.data.op),
                vid: x.data.vid,
            })
            .collect::<Vec<DataRef<'arg, 'dev>>>();
        let rets = ret_shapes
            .into_iter()
            .map(|s| NodeData {
                value: RefCell::new(op.device().new_tensor(s)),
                gradient: RefCell::new(op.device().new_tensor(s)),
            })
            .collect::<Vec<NodeData>>();
        let depth = xs.iter().map(|x| x.data.op.depth).fold(0, cmp::max) + 1;
        OperatorInfo {
            operator: Box::new(op),
            args: args,
            rets: rets,
            forwarded: Cell::new(false),
            depth: depth,
        }
    }
}

fn forward<'arg, 'dev>(op_info: Rc<OperatorInfo<'arg, 'dev>>) {
    let mut reached = HashSet::new();
    let mut queue = VecDeque::new();
    let mut forward_req = BinaryHeap::new();
    queue.push_back(op_info);
    while let Some(op_info) = queue.pop_front() {
        for arg in &op_info.args {
            let ptr = &*arg.op as *const OperatorInfo<'arg, 'dev>;
            if !reached.contains(&ptr) {
                queue.push_back(Rc::clone(&arg.op));
                reached.insert(ptr);
            }
        }
        forward_req.push(Reverse(OperatorInfoCmp(op_info)));
    }
    while let Some(Reverse(OperatorInfoCmp(op_info))) = forward_req.pop() {
        if op_info.forwarded.get() {
            continue;
        }
        op_info.forwarded.set(true);
        let xs = op_info
            .args
            .iter()
            .map(|data| data.op.rets[data.vid].value.borrow())
            .collect::<Vec<Ref<Tensor<'arg>>>>();
        let mut ys = op_info
            .rets
            .iter()
            .map(|ret| ret.value.borrow_mut())
            .collect::<Vec<RefMut<Tensor<'arg>>>>();
        let xs_ref = xs.iter().map(|x| &**x).collect::<Vec<&Tensor<'arg>>>();
        let mut ys_ref = ys
            .iter_mut()
            .map(|y| &mut **y)
            .collect::<Vec<&mut Tensor<'arg>>>();
        op_info.operator.forward(&xs_ref, &mut ys_ref);
    }
}

fn backward<'arg, 'dev>(op_info: Rc<OperatorInfo<'arg, 'dev>>) {
    let mut reached = HashSet::new();
    let mut queue = VecDeque::new();
    let mut backward_req = BinaryHeap::new();
    {
        for ret in &op_info.rets {
            let mut grad = ret.gradient.borrow_mut();
            grad.alloc();
            grad.reset(1.);
        }
    }
    queue.push_back(op_info);
    while let Some(op_info) = queue.pop_front() {
        for arg in &op_info.args {
            let ptr = &*arg.op as *const OperatorInfo<'arg, 'dev>;
            if !reached.contains(&ptr) {
                queue.push_back(Rc::clone(&arg.op));
                reached.insert(ptr);
            }
        }
        backward_req.push(OperatorInfoCmp(op_info));
    }
    while let Some(OperatorInfoCmp(op_info)) = backward_req.pop() {
        let xs = op_info
            .args
            .iter()
            .map(|data| data.op.rets[data.vid].value.borrow())
            .collect::<Vec<Ref<Tensor<'arg>>>>();
        let ys = op_info
            .rets
            .iter()
            .map(|ret| ret.value.borrow())
            .collect::<Vec<Ref<Tensor<'arg>>>>();
        let gys = op_info
            .rets
            .iter()
            .map(|ret| ret.gradient.borrow())
            .collect::<Vec<Ref<Tensor<'arg>>>>();
        let gxs = op_info
            .args
            .iter()
            .map(|data| {
                let mut gx = data.op.rets[data.vid].gradient.borrow_mut();
                if !gx.valid() {
                    gx.alloc();
                    gx.reset(0.);
                }
                &data.op.rets[data.vid].gradient
            })
            .collect::<Vec<&RefCell<Tensor<'arg>>>>();
        let xs_ref = xs.iter().map(|x| &**x).collect::<Vec<&Tensor<'arg>>>();
        let ys_ref = ys.iter().map(|y| &**y).collect::<Vec<&Tensor<'arg>>>();
        let gys_ref = gys.iter().map(|gy| &**gy).collect::<Vec<&Tensor<'arg>>>();
        op_info.operator.backward(&xs_ref, &ys_ref, &gys_ref, &gxs);
    }
}

impl<'arg, 'dev> Node<'arg, 'dev> {
    pub fn create<T: Operator<'arg, 'dev> + 'arg>(
        op: T,
        xs: &[&Node<'arg, 'dev>],
    ) -> Vec<Node<'arg, 'dev>> {
        let op_info = Rc::new(OperatorInfo::new(op, xs));
        (0..op_info.rets.len())
            .map(|i| Node {
                data: DataRef {
                    op: Rc::clone(&op_info),
                    vid: i,
                },
            })
            .collect::<Vec<Node<'arg, 'dev>>>()
    }

    pub fn inner_value(&self) -> Ref<Tensor> {
        self.data.op.rets[self.data.vid].value.borrow()
    }

    pub fn inner_gradient(&self) -> Ref<Tensor> {
        self.data.op.rets[self.data.vid].gradient.borrow()
    }

    pub fn device(&self) -> &'dev Device<'dev> {
        self.data.op.operator.device()
    }
    pub fn forward(&self) {
        forward(Rc::clone(&self.data.op));
    }

    pub fn backward(&self) {
        self.forward();
        backward(Rc::clone(&self.data.op));
    }
}

impl<'arg, 'dev> From<&'arg Tensor<'dev>> for Node<'arg, 'dev> {
    fn from(item: &'arg Tensor<'dev>) -> Self {
        Node::create(op::Input::new(item), &[]).pop().unwrap()
    }
}

impl<'arg, 'dev> From<Tensor<'dev>> for Node<'arg, 'dev> {
    fn from(item: Tensor<'dev>) -> Self {
        Node::create(op::InputOwner::new(item), &[]).pop().unwrap()
    }
}

impl<'arg, 'dev> From<&'arg mut Parameter<'dev>> for Node<'arg, 'dev> {
    fn from(item: &'arg mut Parameter<'dev>) -> Self {
        Node::create(op::Parameter::new(item), &[]).pop().unwrap()
    }
}
