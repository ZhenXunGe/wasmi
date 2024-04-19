use core::cell::RefCell;
use std::rc::Rc;

use parity_wasm::elements::Module;
use wasmi_core::Value;

use crate::{
    isa::Instruction,
    runner::{FunctionContext, InstructionOutcome, ValueStack},
    tracer::Observer,
    Error,
    ModuleRef,
};

pub trait Monitor {
    fn push_element_segment(
        &mut self,
        table_idx: u32,
        type_idx: u32,
        offset_start: u32,
        elements: &[u32],
    ) {
    }

    fn register_module(&mut self, module: &Module, module_ref: &ModuleRef) -> Result<(), Error> {
        Ok(())
    }

    /// Called before each exported function(zkmain or start function) is executed.
    fn invoke_exported_function_pre_hook(&mut self) {}

    /// Called before each instruction is executed.
    fn invoke_instruction_pre_hook(
        &mut self,
        value_stack: &ValueStack,
        function_context: &FunctionContext,
        instruction: &Instruction,
    ) {
    }
    /// Called after each instruction is executed.
    fn invoke_instruction_post_hook(
        &mut self,
        fid: u32,
        iid: u32,
        sp: u32,
        allocated_memory_pages: u32,
        value_stack: &ValueStack,
        function_context: &FunctionContext,
        instruction: &Instruction,
        outcome: &InstructionOutcome,
    ) {
    }

    /// Called after 'call_host' instruction is executed.
    fn invoke_call_host_post_hook(&mut self, return_value: Option<Value>) {}

    #[deprecated]
    fn expose_observer(&self) -> Rc<RefCell<Observer>>;
}
