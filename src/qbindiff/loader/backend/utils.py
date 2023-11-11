# builtin import
from __future__ import annotations
import capstone
from typing import Any, TypeAlias

# local imports
from qbindiff.loader.types import OperandType

# Type aliases
capstoneOperand: TypeAlias = Any  # Relaxed typing


def convert_operand_type(arch: int, cs_operand: capstoneOperand) -> OperandType:
    """
    Function that convert the capstone operand type to Operand type.
    The conversion is specific to the architecture because capstone operand
    type differs from one arch to another.

    Note 1: For the BPF arch, we are yet unsure how to handle the different
            type offered by capstone.
    Note 2: For WASM there is currently not documentation on the different
            operand so it is hard to convert them.
    Note 3: There is actually no operands for the EVM arch.
    """

    operands = {
        capstone.CS_ARCH_ARM: {
            capstone.arm_const.ARM_OP_INVALID: OperandType.unknown,
            capstone.arm_const.ARM_OP_REG: OperandType.register,
            capstone.arm_const.ARM_OP_IMM: OperandType.immediate,
            capstone.arm_const.ARM_OP_MEM: OperandType.memory,
            capstone.arm_const.ARM_OP_FP: OperandType.float_point,
            capstone.arm_const.ARM_OP_CIMM: OperandType.coprocessor,
            capstone.arm_const.ARM_OP_PIMM: OperandType.coprocessor,
            capstone.arm_const.ARM_OP_SETEND: OperandType.arm_setend,
            capstone.arm_const.ARM_OP_SYSREG: OperandType.coprocessor,
        },
        capstone.CS_ARCH_ARM64: {
            capstone.arm64_const.ARM64_OP_INVALID: OperandType.unknown,
            capstone.arm64_const.ARM64_OP_REG: OperandType.register,
            capstone.arm64_const.ARM64_OP_IMM: OperandType.immediate,
            capstone.arm64_const.ARM64_OP_MEM: OperandType.memory,
            capstone.arm64_const.ARM64_OP_FP: OperandType.float_point,
            capstone.arm64_const.ARM64_OP_CIMM: OperandType.coprocessor,
            capstone.arm64_const.ARM64_OP_REG_MRS: OperandType.coprocessor,
            capstone.arm64_const.ARM64_OP_REG_MSR: OperandType.coprocessor,
            capstone.arm64_const.ARM64_OP_PSTATE: OperandType.register,
            capstone.arm64_const.ARM64_OP_SYS: OperandType.arm_memory_management,
            capstone.arm64_const.ARM64_OP_SVCR: OperandType.coprocessor,
            capstone.arm64_const.ARM64_OP_PREFETCH: OperandType.arm_memory_management,
            capstone.arm64_const.ARM64_OP_BARRIER: OperandType.arm_memory_management,
            capstone.arm64_const.ARM64_OP_SME_INDEX: OperandType.arm_sme,
        },
        capstone.CS_ARCH_MIPS: {
            capstone.mips_const.MIPS_OP_INVALID: OperandType.unknown,
            capstone.mips_const.MIPS_OP_REG: OperandType.register,
            capstone.mips_const.MIPS_OP_IMM: OperandType.immediate,
            capstone.mips_const.MIPS_OP_MEM: OperandType.memory,
        },
        capstone.CS_ARCH_X86: {
            capstone.x86_const.X86_OP_INVALID: OperandType.unknown,
            capstone.x86_const.X86_OP_REG: OperandType.register,
            capstone.x86_const.X86_OP_IMM: OperandType.immediate,
            capstone.x86_const.X86_OP_MEM: OperandType.memory,
        },
        capstone.CS_ARCH_PPC: {
            capstone.ppc_const.PPC_OP_INVALID: OperandType.unknown,
            capstone.ppc_const.PPC_OP_REG: OperandType.register,
            capstone.ppc_const.PPC_OP_IMM: OperandType.immediate,
            capstone.ppc_const.PPC_OP_MEM: OperandType.memory,
            capstone.ppc_const.PPC_OP_CRX: OperandType.register,
        },
        capstone.CS_ARCH_SPARC: {
            capstone.sparc_const.SPARC_OP_INVALID: OperandType.unknown,
            capstone.sparc_const.SPARC_OP_REG: OperandType.register,
            capstone.sparc_const.SPARC_OP_IMM: OperandType.immediate,
            capstone.sparc_const.SPARC_OP_MEM: OperandType.memory,
        },
        capstone.CS_ARCH_SYSZ: {
            capstone.sysz_const.SYSZ_OP_INVALID: OperandType.unknown,
            capstone.sysz_const.SYSZ_OP_REG: OperandType.register,
            capstone.sysz_const.SYSZ_OP_IMM: OperandType.immediate,
            capstone.sysz_const.SYSZ_OP_MEM: OperandType.memory,
            capstone.sysz_const.SYSZ_OP_ACREG: OperandType.register,
        },
        capstone.CS_ARCH_XCORE: {
            capstone.xcore_const.XCORE_OP_INVALID: OperandType.unknown,
            capstone.xcore_const.XCORE_OP_REG: OperandType.register,
            capstone.xcore_const.XCORE_OP_IMM: OperandType.immediate,
            capstone.xcore_const.XCORE_OP_MEM: OperandType.memory,
        },
        capstone.CS_ARCH_M68K: {
            capstone.m68k_const.M68K_OP_INVALID: OperandType.unknown,
            capstone.m68k_const.M68K_OP_REG: OperandType.register,
            capstone.m68k_const.M68K_OP_IMM: OperandType.immediate,
            capstone.m68k_const.M68K_OP_MEM: OperandType.memory,
            capstone.m68k_const.M68K_OP_FP_SINGLE: OperandType.float_point,
            capstone.m68k_const.M68K_OP_FP_DOUBLE: OperandType.float_point,
            capstone.m68k_const.M68K_OP_REG_BITS: OperandType.register,
            capstone.m68k_const.M68K_OP_REG_PAIR: OperandType.register,
            capstone.m68k_const.M68K_OP_BR_DISP: OperandType.memory,
        },
        capstone.CS_ARCH_TMS320C64X: {
            capstone.tms320c64x_const.TMS320C64X_OP_INVALID: OperandType.unknown,
            capstone.tms320c64x_const.TMS320C64X_OP_REG: OperandType.register,
            capstone.tms320c64x_const.TMS320C64X_OP_IMM: OperandType.immediate,
            capstone.tms320c64x_const.TMS320C64X_OP_MEM: OperandType.memory,
            capstone.tms320c64x_const.TMS320C64X_OP_REGPAIR: OperandType.register,
        },
        capstone.CS_ARCH_M680X: {
            capstone.m680x_const.M680X_OP_INVALID: OperandType.unknown,
            capstone.m680x_const.M680X_OP_REGISTER: OperandType.register,
            capstone.m680x_const.M680X_OP_IMMEDIATE: OperandType.immediate,
            capstone.m680x_const.M680X_OP_INDEXED: OperandType.memory,
            capstone.m680x_const.M680X_OP_EXTENDED: OperandType.memory,
            capstone.m680x_const.M680X_OP_DIRECT: OperandType.memory,
            capstone.m680x_const.M680X_OP_RELATIVE: OperandType.memory,
            capstone.m680x_const.M680X_OP_CONSTANT: OperandType.immediate,
        },
        capstone.CS_ARCH_MOS65XX: {
            capstone.mos65xx_const.MOS65XX_OP_INVALID: OperandType.unknown,
            capstone.mos65xx_const.MOS65XX_OP_REG: OperandType.register,
            capstone.mos65xx_const.MOS65XX_OP_IMM: OperandType.immediate,
            capstone.mos65xx_const.MOS65XX_OP_MEM: OperandType.memory,
        },
        capstone.CS_ARCH_WASM: {
            capstone.wasm_const.WASM_OP_INVALID: OperandType.unknown,
            capstone.wasm_const.WASM_OP_NONE: OperandType.unknown,
            capstone.wasm_const.WASM_OP_INT7: OperandType.unknown,
            capstone.wasm_const.WASM_OP_VARUINT32: OperandType.unknown,
            capstone.wasm_const.WASM_OP_VARUINT64: OperandType.unknown,
            capstone.wasm_const.WASM_OP_UINT32: OperandType.unknown,
            capstone.wasm_const.WASM_OP_UINT64: OperandType.unknown,
            capstone.wasm_const.WASM_OP_IMM: OperandType.immediate,
            capstone.wasm_const.WASM_OP_BRTABLE: OperandType.unknown,
        },
        capstone.CS_ARCH_BPF: {
            capstone.bpf_const.BPF_OP_INVALID: OperandType.unknown,
            capstone.bpf_const.BPF_OP_REG: OperandType.register,
            capstone.bpf_const.BPF_OP_IMM: OperandType.immediate,
            capstone.bpf_const.BPF_OP_OFF: OperandType.unknown,
            capstone.bpf_const.BPF_OP_MEM: OperandType.memory,
            capstone.bpf_const.BPF_OP_MMEM: OperandType.unknown,
            capstone.bpf_const.BPF_OP_MSH: OperandType.unknown,
            capstone.bpf_const.BPF_OP_EXT: OperandType.unknown,
        },
        capstone.CS_ARCH_RISCV: {
            capstone.riscv_const.RISCV_OP_INVALID: OperandType.unknown,
            capstone.riscv_const.RISCV_OP_REG: OperandType.register,
            capstone.riscv_const.RISCV_OP_IMM: OperandType.immediate,
            capstone.riscv_const.RISCV_OP_MEM: OperandType.memory,
        },
        capstone.CS_ARCH_SH: {
            capstone.sh_const.SH_OP_INVALID: OperandType.unknown,
            capstone.sh_const.SH_OP_REG: OperandType.register,
            capstone.sh_const.SH_OP_IMM: OperandType.immediate,
            capstone.sh_const.SH_OP_MEM: OperandType.memory,
        },
        capstone.CS_ARCH_TRICORE: {
            capstone.tricore_const.TRICORE_OP_INVALID: OperandType.unknown,
            capstone.tricore_const.TRICORE_OP_REG: OperandType.register,
            capstone.tricore_const.TRICORE_OP_IMM: OperandType.immediate,
            capstone.tricore_const.TRICORE_OP_MEM: OperandType.memory,
        },
        capstone.CS_ARCH_EVM: {
            # No operands are defined by capstone
        },
    }

    arch_specific_operands = operands.get(arch)
    if not arch_specific_operands:
        raise NotImplementedError(f"Unrecognized capstone arch {arch}")
    operand_type = arch_specific_operands.get(cs_operand.type)
    if not operand_type:
        raise Exception(f"Unrecognized capstone operand {cs_operand.type} for arch {arch}")
    return operand_type
