use wasmi_core::Value;

use crate::{
    isa::Instruction,
    runner::{FunctionContext, ValueStack},
    tracer::etable::RunInstructionTracePre,
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

    fn register_module(
        &mut self,
        module: &parity_wasm::elements::Module,
        module_ref: &ModuleRef,
    ) -> Result<(), Error> {
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
    ) {
    }

    /// Called before 'return' instruction is executed.
    fn invoke_return_pre_hook(&mut self) {}
    /// Called before 'call' instruction is executed.
    fn invoke_call_pre_hook(&mut self, function_index: u32, instruction_index: u32) {}
    /// Called after 'call_host' instruction is executed.
    fn invoke_call_host_post_hook(&mut self, return_value: Option<Value>) {}
}
