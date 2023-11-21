use std::ops::Add;
use std::ops::Mul;

use num_bigint::BigUint;

pub mod br_table;
pub mod frame_table;
pub mod image_table;
pub mod init_memory_table;
pub mod instruction_table;
pub mod memory_table;
pub mod opcode;

pub(crate) const COMMON_RANGE_OFFSET: u32 = 32;

pub trait FromBn: Sized + Add<Self, Output = Self> + Mul<Self, Output = Self> {
    fn zero() -> Self;
    fn from_bn(bn: &BigUint) -> Self;
}

impl FromBn for BigUint {
    fn zero() -> Self {
        BigUint::from(0u64)
    }

    fn from_bn(bn: &BigUint) -> Self {
        bn.clone()
    }
}

