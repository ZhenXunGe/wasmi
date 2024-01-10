use core::{cell::RefCell, fmt::Debug};
use std::{collections::HashMap, rc::Rc, sync::Arc};

use specs::{
    brtable::{ElemEntry, ElemTable, BrTable},
    configure_table::ConfigureTable,
    etable::EventTable,
    host_function::HostFunctionDesc,
    imtable::InitMemoryTable,
    itable::{InstructionTable, InstructionTableEntry},
    jtable::{JumpTable, StaticFrameEntry, STATIC_FRAME_ENTRY_NUMBER},
    mtable::VarType,
    state::{InitializationState, UpdateCompilationTable},
    types::FunctionType,
    CompilationTable,
    ExecutionTable,
    Tables,
};

use crate::{
    func::FuncInstanceInternal,
    runner::{from_value_internal_to_u64_with_typ, ValueInternal},
    FuncRef,
    GlobalRef,
    MemoryRef,
    Module,
    ModuleRef,
    Signature,
    DEFAULT_VALUE_STACK_LIMIT,
};

use self::{etable::ETable, imtable::IMTable, phantom::PhantomFunction};

pub mod etable;
pub mod imtable;
pub mod phantom;

#[derive(Debug)]
pub struct FuncDesc {
    pub index_within_jtable: u32,
    pub ftype: FunctionType,
    pub signature: Signature,
}

pub trait SliceDumper {
    fn dump(&mut self, tables: Tables);
    fn get_capacity(&self) -> usize;
    fn dump_enabled(&self) -> bool;
}

impl Debug for dyn SliceDumper {
    fn fmt(&self, _: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Ok(())
    }
}

#[derive(Debug)]
pub struct Tracer {
    pub itable: InstructionTable,
    pub imtable: IMTable,
    pub br_table: BrTable,
    pub etable: EventTable,
    pub jtable: JumpTable,
    pub elem_table: ElemTable,
    pub configure_table: ConfigureTable,
    type_of_func_ref: Vec<(FuncRef, u32)>,
    function_lookup: Vec<(FuncRef, u32)>,
    pub(crate) last_jump_eid: Vec<u32>,
    function_index_allocator: u32,
    pub(crate) function_index_translation: HashMap<u32, FuncDesc>,
    pub host_function_index_lookup: HashMap<usize, HostFunctionDesc>,
    pub static_jtable_entries: Vec<StaticFrameEntry>,
    pub phantom_functions: Vec<String>,
    pub phantom_functions_ref: Vec<FuncRef>,
    // Wasm Image Function Idx
    pub wasm_input_func_idx: Option<u32>,
    pub wasm_input_func_ref: Option<FuncRef>,
    // wasm perf opt by caching
    itable_entries: HashMap<u64, InstructionTableEntry>,
    function_map: HashMap<usize, u32>,
    host_function_map: HashMap<usize, u32>,
    // continuation
    witness_dumper: Rc<RefCell<dyn SliceDumper>>,
    fid_of_entry: u32,
    prev_eid: u32,
    cur_imtable: InitMemoryTable,
    cur_state: InitializationState<u32>,
}

impl Tracer {
    /// Create an empty tracer
    pub fn new(
        host_plugin_lookup: HashMap<usize, HostFunctionDesc>,
        phantom_functions: &Vec<String>,
        witness_dumper: Rc<RefCell<dyn SliceDumper>>,
    ) -> Self {
        Tracer {
            itable: InstructionTable::default(),
            imtable: IMTable::default(),
            br_table: BrTable::default(),
            etable: EventTable::default(),
            last_jump_eid: vec![],
            jtable: JumpTable::default(),
            elem_table: ElemTable::default(),
            configure_table: ConfigureTable::default(),
            type_of_func_ref: vec![],
            function_lookup: vec![],
            function_index_allocator: 1,
            function_index_translation: Default::default(),
            host_function_index_lookup: host_plugin_lookup,
            static_jtable_entries: vec![],
            phantom_functions: phantom_functions.clone(),
            phantom_functions_ref: vec![],
            wasm_input_func_ref: None,
            wasm_input_func_idx: None,
            itable_entries: HashMap::new(),
            function_map: HashMap::new(),
            host_function_map: HashMap::new(),
            witness_dumper,
            // #[cfg(feature="continuation")]
            fid_of_entry: 0, // change when initializing module
            prev_eid: 0,
            cur_imtable: InitMemoryTable::default(),
            cur_state: InitializationState::default(), // change when setting fid_of_entry
        }
    }

    pub fn push_frame(&mut self) {
        self.last_jump_eid.push(self.etable.get_latest_eid());
    }

    pub fn pop_frame(&mut self) {
        self.last_jump_eid.pop().unwrap();
    }

    pub fn last_jump_eid(&self) -> u32 {
        *self.last_jump_eid.last().unwrap()
    }

    pub fn eid(&self) -> u32 {
        self.etable.get_latest_eid()
    }

    fn allocate_func_index(&mut self) -> u32 {
        let r = self.function_index_allocator;
        self.function_index_allocator = r + 1;
        r
    }

    fn lookup_host_plugin(&self, function_index: usize) -> HostFunctionDesc {
        self.host_function_index_lookup
            .get(&function_index)
            .unwrap()
            .clone()
    }
}

impl Tracer {
    pub(crate) fn dump_witness(&mut self, is_last_slice: bool) {
        // keep etable eid
        self.prev_eid = self.eid() - 1;
        let mut etable = std::mem::take(&mut self.etable);
        let etable_entires = etable.entries_mut();
        // If it is not the last slice, push a step to keep eid correct.
        if !is_last_slice {
            let last_entry = etable_entires.last().unwrap().clone();
            self.etable = EventTable::new(vec![last_entry])
        }

        let static_jtable = Arc::new(
                self
                .static_jtable_entries
                .clone()
                .try_into()
                .expect(&format!(
                    "The number of static frame entries should be {}",
                    STATIC_FRAME_ENTRY_NUMBER
                )),
        );

        let br_table = Arc::new(self.br_table.clone());

        let compilation_tables = CompilationTable {
            itable: Arc::new(self.itable.clone()),
            imtable: self.cur_imtable.clone(),
            br_table: br_table.clone(),
            elem_table: Arc::new(self.elem_table.clone()),
            configure_table: Arc::new(self.configure_table),
            static_jtable: Arc::clone(&static_jtable),
            initialization_state: self.cur_state.clone(),
        };

        // update current state
        self.cur_state =
            compilation_tables.update_initialization_state(etable_entires, is_last_slice);

        // update current memory table
        // If it is not the last slice, push a helper step to get the post initialization state.
        if !is_last_slice {
            etable_entires.pop();
        }
        self.cur_imtable = compilation_tables.update_init_memory_table(etable_entires);

        let post_image_table = CompilationTable {
            itable: Arc::new(self.itable.clone()),
            imtable: self.cur_imtable.clone(),
            br_table,
            elem_table: Arc::new(self.elem_table.clone()),
            configure_table: Arc::new(self.configure_table),
            static_jtable: static_jtable,
            initialization_state: self.cur_state.clone(),
        };

        let execution_tables = ExecutionTable {
            etable,
            jtable: Arc::new(self.jtable.clone()),
        };

        self.witness_dumper.borrow_mut().dump(Tables {
            compilation_tables,
            execution_tables,
            post_image_table,
            is_last_slice,
        });
    }

    pub(crate) fn get_prev_eid(&self) -> u32 {
        self.prev_eid
    }

    pub(crate) fn slice_capability(&self) -> u32 {
        self.witness_dumper.borrow().get_capacity() as u32
    }

    pub fn dump_enabled(&self) -> bool {
        self.witness_dumper.borrow().dump_enabled()
    }

    pub(crate) fn set_fid_of_entry(&mut self, fid_of_entry: u32) {
        self.fid_of_entry = fid_of_entry;

        self.cur_state = InitializationState {
            eid: 1,
            fid: fid_of_entry,
            iid: 0,
            frame_id: 0,
            sp: DEFAULT_VALUE_STACK_LIMIT as u32 - 1,

            host_public_inputs: 1,
            context_in_index: 1,
            context_out_index: 1,
            external_host_call_call_index: 1,

            initial_memory_pages: self.configure_table.init_memory_pages,
            maximal_memory_pages: self.configure_table.maximal_memory_pages,
            #[cfg(feature = "continuation")]
            jops: 0,
        };
    }

    pub fn get_fid_of_entry(&self) -> u32 {
        self.fid_of_entry
    }
}

impl Tracer {
    pub(crate) fn push_init_memory(&mut self, memref: MemoryRef) {
        // one page contains 64KB/8 = 64*1024/8=8192 u64 entries
        const ENTRIES: u32 = 8192;

        let pages = (*memref).limits().initial();
        for i in 0..(pages * ENTRIES) {
            let mut buf = [0u8; 8];
            (*memref).get_into(i * 8, &mut buf).unwrap();

            let v = u64::from_le_bytes(buf);

            if v != 0 {
                self.imtable
                    .push(false, true, i, VarType::I64, u64::from_le_bytes(buf));
            }
        }

        // update current memory table
        self.cur_imtable = self.imtable.finalized();
        self.br_table = self.itable.create_brtable();
    }

    pub(crate) fn push_global(&mut self, globalidx: u32, globalref: &GlobalRef) {
        let vtype = globalref.elements_value_type().into();

        self.imtable.push(
            true,
            globalref.is_mutable(),
            globalidx,
            vtype,
            from_value_internal_to_u64_with_typ(vtype, ValueInternal::from(globalref.get())),
        );
    }

    pub(crate) fn push_elem(&mut self, table_idx: u32, offset: u32, func_idx: u32, type_idx: u32) {
        self.elem_table.insert(ElemEntry {
            table_idx,
            type_idx,
            offset,
            func_idx,
        })
    }

    pub(crate) fn push_type_of_func_ref(&mut self, func: FuncRef, type_idx: u32) {
        self.type_of_func_ref.push((func, type_idx))
    }

    #[allow(dead_code)]
    pub(crate) fn statistics_instructions<'a>(&mut self, module_instance: &ModuleRef) {
        let mut func_index = 0;
        let mut insts = vec![];

        loop {
            if let Some(func) = module_instance.func_by_index(func_index) {
                let body = func.body().unwrap();

                let code = &body.code.vec;

                for inst in code {
                    if insts.iter().position(|i| i == inst).is_none() {
                        insts.push(inst.clone())
                    }
                }
            } else {
                break;
            }

            func_index = func_index + 1;
        }

        for inst in insts {
            println!("{:?}", inst);
        }
    }

    pub(crate) fn lookup_type_of_func_ref(&self, func_ref: &FuncRef) -> u32 {
        self.type_of_func_ref
            .iter()
            .find(|&f| f.0 == *func_ref)
            .unwrap()
            .1
    }

    pub(crate) fn register_module_instance(
        &mut self,
        module: &Module,
        module_instance: &ModuleRef,
    ) {
        let start_fn_idx = module.module().start_section();

        {
            let mut func_index = 0;

            loop {
                if let Some(func) = module_instance.func_by_index(func_index) {
                    if Some(&func) == self.wasm_input_func_ref.as_ref() {
                        self.wasm_input_func_idx = Some(func_index)
                    }

                    let func_index_in_itable = if Some(func_index) == start_fn_idx {
                        0
                    } else {
                        self.allocate_func_index()
                    };

                    let ftype = match *func.as_internal() {
                        crate::func::FuncInstanceInternal::Internal { .. } => {
                            FunctionType::WasmFunction
                        }
                        crate::func::FuncInstanceInternal::Host {
                            host_func_index, ..
                        } => {
                            let plugin_desc = self.lookup_host_plugin(host_func_index);

                            match plugin_desc {
                                HostFunctionDesc::Internal {
                                    name,
                                    op_index_in_plugin,
                                    plugin,
                                } => FunctionType::HostFunction {
                                    plugin,
                                    function_index: host_func_index,
                                    function_name: name,
                                    op_index_in_plugin,
                                },
                                HostFunctionDesc::External { name, op, sig } => {
                                    FunctionType::HostFunctionExternal {
                                        function_name: name,
                                        op,
                                        sig,
                                    }
                                }
                            }
                        }
                    };

                    self.function_lookup
                        .push((func.clone(), func_index_in_itable));

                    match *func.as_internal() {
                        FuncInstanceInternal::Internal {
                            image_func_index, ..
                        } => {
                            self.function_map
                                .insert(image_func_index, func_index_in_itable);
                        }
                        FuncInstanceInternal::Host {
                            host_func_index, ..
                        } => {
                            self.host_function_map
                                .insert(host_func_index, func_index_in_itable);
                        }
                    }

                    self.function_index_translation.insert(
                        func_index,
                        FuncDesc {
                            index_within_jtable: func_index_in_itable,
                            ftype,
                            signature: func.signature().clone(),
                        },
                    );
                    func_index = func_index + 1;
                } else {
                    break;
                }
            }
        }

        {
            let phantom_functions_ref = self.phantom_functions.clone();

            for func_name in phantom_functions_ref {
                let func = module_instance.func_by_name(&func_name).unwrap();

                self.push_phantom_function(func);
            }
        }

        {
            let mut func_index = 0;

            loop {
                if let Some(func) = module_instance.func_by_index(func_index) {
                    let funcdesc = self.function_index_translation.get(&func_index).unwrap();

                    if self.is_phantom_function(&func) {
                        let instructions = PhantomFunction::build_phantom_function_instructions(
                            &funcdesc.signature,
                            self.wasm_input_func_idx.unwrap(),
                        );

                        for (iid, inst) in instructions.into_iter().enumerate() {
                            self.itable.push(
                                funcdesc.index_within_jtable,
                                iid as u32,
                                inst.into(&self.function_index_translation),
                            )
                        }
                    } else {
                        if let Some(body) = func.body() {
                            let code = &body.code;
                            let mut iter = code.iterate_from(0);
                            loop {
                                let pc = iter.position();
                                if let Some(instruction) = iter.next() {
                                    let _ = self.itable.push(
                                        funcdesc.index_within_jtable,
                                        pc,
                                        instruction.into(&self.function_index_translation),
                                    );
                                    self.itable_entries.insert(
                                        ((funcdesc.index_within_jtable as u64) << 32) + pc as u64,
                                        self.itable.entries().last().unwrap().clone(),
                                    );
                                } else {
                                    break;
                                }
                            }
                        }
                    }

                    func_index = func_index + 1;
                } else {
                    break;
                }
            }
        }
    }

    pub fn lookup_function(&self, function: &FuncRef) -> u32 {
        match *function.as_internal() {
            FuncInstanceInternal::Internal {
                image_func_index, ..
            } => *self.function_map.get(&image_func_index).unwrap(),
            FuncInstanceInternal::Host {
                host_func_index, ..
            } => *self.host_function_map.get(&host_func_index).unwrap(),
        }
    }

    pub fn lookup_ientry(&self, function: &FuncRef, pos: u32) -> InstructionTableEntry {
        let function_idx = self.lookup_function(function);

        let key = ((function_idx as u64) << 32) + pos as u64;
        self.itable_entries.get(&key).unwrap().clone()

        // unreachable!()
    }

    pub fn lookup_first_inst(&self, function: &FuncRef) -> InstructionTableEntry {
        let function_idx = self.lookup_function(function);

        for ientry in self.itable.entries() {
            if ientry.fid == function_idx {
                return ientry.clone();
            }
        }

        unreachable!();
    }

    pub fn push_phantom_function(&mut self, function: FuncRef) {
        self.phantom_functions_ref.push(function)
    }

    pub fn is_phantom_function(&self, func: &FuncRef) -> bool {
        self.phantom_functions_ref.contains(func)
    }
}
