"""
Testing code for shared memory allocation for CUDA devices.

The names of the fields in a the shared memory structure are from the use of these functions
in a DNA optimization application.

"""
import os
from enum import auto, IntEnum
import logging
from typing import Tuple

# Must be before the first import of numba
os.environ['NUMBA_ENABLE_CUDASIM'] = '0'

import numba
from numba import cuda
from shared_memory_struct import SharedMemoryStructCalculator
from shared_memory_struct import SHARED_MEMORY_WORD_SIZE, NUMBER_SHARED_MEMORY_BANKS

import numpy as np

logging.getLogger("numba").setLevel(logging.DEBUG)


class Fields(IntEnum):
    F32: int = 0
    U32: int = auto()
    I32: int = auto()


# for m in Fields.__members__.values():
#     globals()[m.name] = m.value
    

class FieldsWith64(IntEnum):
    F32: int = 0
    F64: int = auto()
    U32: int = auto()
    I32: int = auto()


# Element types
F32_DTYPE = numba.float32
F64_DTYPE = numba.float64
U32_DTYPE = numba.uint32
I32_DTYPE = numba.int32


# NUMBA_INDICE_TYPE = numba.uint32
NUMBA_INDICE_TYPE = numba.uint8


class SMFields(IntEnum):
    BEST_FIDELITY_OF_SEQUENCES = 0
    BEST_COMBINATION_FOR_SEQUENCES = auto()
    LIGATION_PROBABILITIES = auto()
    CURRENT_COMBINATION_INDICES = auto()
    CURRENT_COMBINATION_AND_RESERVED = auto()
    CURRENT_COMBINATION_AND_RESERVED_REV_COMP = auto()
    WORKING_LIGATIONS_FOR_4MER = auto()


# Types of the arrays in shared memory; use these definitions to prevent errors:
BEST_FIDELITY_OF_SEQUENCES_DTYPE = numba.float32
BEST_COMBINATION_FOR_SEQUENCES_DTYPE = numba.uint32
LIGATION_PROBABILITIES_DTYPE = numba.float32
CURRENT_COMBINATION_INDICES_DTYPE = NUMBA_INDICE_TYPE
CURRENT_COMBINATION_AND_RESERVED_DTYPE = NUMBA_INDICE_TYPE
CURRENT_COMBINATION_AND_RESERVED_REV_COMP_DTYPE = NUMBA_INDICE_TYPE
WORKING_LIGATIONS_FOR_4MER_DTYPE = numba.float32

number_selected_overhangs = 8
num_overhangs = 19
fidelity_table_row_length = 256

struct_field_descriptions_optimization = [
        (SMFields.BEST_FIDELITY_OF_SEQUENCES, BEST_FIDELITY_OF_SEQUENCES_DTYPE, 1),
        (SMFields.BEST_COMBINATION_FOR_SEQUENCES, BEST_COMBINATION_FOR_SEQUENCES_DTYPE, 1),
        (SMFields.LIGATION_PROBABILITIES, LIGATION_PROBABILITIES_DTYPE, number_selected_overhangs),
        (SMFields.CURRENT_COMBINATION_INDICES, CURRENT_COMBINATION_INDICES_DTYPE, number_selected_overhangs),
        (SMFields.CURRENT_COMBINATION_AND_RESERVED, CURRENT_COMBINATION_AND_RESERVED_DTYPE, num_overhangs),
        (SMFields.CURRENT_COMBINATION_AND_RESERVED_REV_COMP, CURRENT_COMBINATION_AND_RESERVED_REV_COMP_DTYPE,
         num_overhangs),
        (SMFields.WORKING_LIGATIONS_FOR_4MER, WORKING_LIGATIONS_FOR_4MER_DTYPE, fidelity_table_row_length)
]


#  These functions are the same as in shared_memory_struct.py but modified for readability and debugging the bank size
@cuda.jit(device=True)
def slice_of_shared_memory(field_index, struct_info, thread_index):
    # Offset the shared memory for each thread to help distribute across shared memory banks
    struct_size = struct_info[0]
    itemsize, offset_in_items, plurality = struct_info[1][numba.int32(field_index)][:]
    offset_to_start = thread_index * struct_size
    first_index = offset_to_start // itemsize + offset_in_items
    bank = ((first_index * itemsize) // SHARED_MEMORY_WORD_SIZE) % NUMBER_SHARED_MEMORY_BANKS
    print(thread_index, itemsize, offset_in_items, plurality, first_index, first_index * itemsize, bank)
    return slice(first_index, first_index + plurality)


@cuda.jit(device=True)
def slice_of_shared_memory_across_threads(field_index, struct_info, number_threads):
    # Only works for 1D fields
    field_index = numba.int32(field_index)
    first_index = struct_info[1][field_index][1]
    return slice(first_index, first_index + (struct_info[0] * number_threads) // struct_info[1][field_index][0],
                 struct_info[0] // struct_info[1][field_index][0])


@cuda.jit
def run_optimization_allocations(struct_info):
    thread_id = cuda.threadIdx.x
    
    if thread_id == 0:
        print('best_fidelity_per_sequence')
    
    best_fidelity_per_sequence = cuda.shared.array(0, dtype=BEST_FIDELITY_OF_SEQUENCES_DTYPE)[
        slice_of_shared_memory(SMFields.BEST_FIDELITY_OF_SEQUENCES.value, struct_info, thread_id)]
    if thread_id == 0:
        print('best_combination_index_for_block')

    best_combination_index_for_block = cuda.shared.array(0, dtype=BEST_COMBINATION_FOR_SEQUENCES_DTYPE)[
        slice_of_shared_memory(SMFields.BEST_COMBINATION_FOR_SEQUENCES.value, struct_info, thread_id)]
    if thread_id == 0:
        print('ligation_probabilities')

    ligation_probabilities = cuda.shared.array(0, dtype=LIGATION_PROBABILITIES_DTYPE)[
        slice_of_shared_memory(SMFields.LIGATION_PROBABILITIES.value, struct_info, thread_id)]
    if thread_id == 0:
        print('current_combination_indices')
    current_combination_indices = cuda.shared.array(0, dtype=CURRENT_COMBINATION_INDICES_DTYPE)[
        slice_of_shared_memory(SMFields.CURRENT_COMBINATION_INDICES.value, struct_info, thread_id)]
    if thread_id == 0:
        print('current_combination_and_reserved')
    current_combination_and_reserved = cuda.shared.array(0, dtype=CURRENT_COMBINATION_AND_RESERVED_DTYPE)[
        slice_of_shared_memory(SMFields.CURRENT_COMBINATION_AND_RESERVED.value, struct_info, thread_id)]
    if thread_id == 0:
        print('current_combination_and_reserved_rev_comp')
    current_combination_and_reserved_rev_comp = \
    cuda.shared.array(0, dtype=CURRENT_COMBINATION_AND_RESERVED_REV_COMP_DTYPE)[
        slice_of_shared_memory(SMFields.CURRENT_COMBINATION_AND_RESERVED_REV_COMP.value, struct_info, thread_id)]
    if thread_id == 0:
        print('working_ligations_for_4mer')
    working_ligations_for_4mer = cuda.shared.array(0, dtype=WORKING_LIGATIONS_FOR_4MER_DTYPE)[
        slice_of_shared_memory(SMFields.WORKING_LIGATIONS_FOR_4MER.value, struct_info, thread_id)]


@cuda.jit
def run_with_small_arrays(struct_info: Tuple):
            
    t_id = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    number_threads = cuda.blockDim.x
    if t_id == 1 and bx == 0:
        # from pdb import set_trace; set_trace()
        print('threadid:', t_id)
        print('blockIdx:', bx)
        print('cuda.blockDim.x:', cuda.blockDim.x)
        
    # if t_id == 31:
    #     f32_arr = cuda.shared.array(0, dtype=F32_DTYPE)[slice_of_shared_memory(Fields.F32, struct_info, t_id)]

    f32_arr = cuda.shared.array(0, dtype=F32_DTYPE)[slice_of_shared_memory(Fields.F32.value, struct_info, t_id)]
    # # f64_arr = cuda.shared.array(0, dtype=F64_DTYPE)[slice_of_shared_memory(Fields.F64, struct_info, t_id)]
    ui32_arr = cuda.shared.array(0, dtype=U32_DTYPE)[slice_of_shared_memory(Fields.U32.value, struct_info, t_id)]
    i32_arr = cuda.shared.array(0, dtype=I32_DTYPE)[slice_of_shared_memory(Fields.I32.value, struct_info, t_id)]


    # ui32_arr_threads = cuda.shared.array(0, dtype=U32_DTYPE)[slice_of_shared_memory_across_threads(U32, struct_info)]

    for i in range(0, len(ui32_arr)):
        ui32_arr[i] = np.uint32(42 + i + t_id)

    f32_arr[0] = 3.14 + np.float32(t_id)
    # for i in range(0, len(f64_arr)):
    #     f64_arr[i] = numba.float64(100. + t_id + i)
    # if True:  # t_id == 1:
    #     print(t_id, 'f64_arr', f64_arr[0], f64_arr[1], f64_arr[2], f64_arr[3])
    
    i32_arr[0] = 1 + np.int32(t_id)
    print(t_id, 'f32_arr', f32_arr[0])
    
    # for i in range(0, len(f64_arr)):
    #     print(t_id, 'f64_arr', i, f64_arr[i])
    print(t_id, 'ui32_arr', ui32_arr[0], ui32_arr[1], ui32_arr[2], ui32_arr[3], ui32_arr[4], ui32_arr[5])
    print(t_id, 'i32_arr', i32_arr[0])
    
    cuda.syncthreads()
    if t_id == 0:
        f32_arr_threads = cuda.shared.array(0, dtype=F32_DTYPE)[slice_of_shared_memory_across_threads(Fields.F32.value, struct_info, number_threads)]
        print(t_id, 'f32_arr_threads', len(f32_arr_threads))
        for x in f32_arr_threads:
            print(x)
        u32_arr_threads = cuda.shared.array(0, dtype=U32_DTYPE)[slice_of_shared_memory_across_threads(Fields.U32.value, struct_info, number_threads)]
        print(t_id, 'u32_arr_threads', len(u32_arr_threads))
        for x in u32_arr_threads:
            print(x)
        print("")


def check_bank_allocation_for_ligation_optimization():
    number_threads_per_block = 32
    struct_fields = SharedMemoryStructCalculator()
    struct_fields.add_fields(struct_field_descriptions_optimization)
    
    print(f"struct_fields.total_size_per_thread = {struct_fields.struct_size()}")
    print(f"struct_fields.total_size_for_all_threads() = {struct_fields.struct_size_for_all_threads(number_threads_per_block)}")
    print(f"offsets = {struct_fields.field_info()}")
    
    struct_info = (struct_fields.struct_size(), cuda.to_device(np.array(struct_fields.field_info())))
    print(f"Total shared memory size: {struct_fields.struct_size_for_all_threads(number_threads_per_block)}")
    # Launch the kernel
    kernel_parameters = (struct_info,)
    # launch_parameters: [grid_dim, block_dim, stream, dyn_shared_mem_size]
    launch_parameters = (1, number_threads_per_block, 0, struct_fields.struct_size_for_all_threads(number_threads_per_block))
    run_optimization_allocations[launch_parameters](*kernel_parameters)
    
    cuda.synchronize()


def check_bank_allocation_for_small_arrays():
  
    number_threads_per_block = 32
    shared_memory_word_size = 4
    number_shared_memory_banks = 32
    struct_fields = SharedMemoryStructCalculator(shared_memory_word_size, number_shared_memory_banks)
    # Field id, dtype, plurality
    if shared_memory_word_size == 8:
        # struct_field_descriptions = (
        #         (FieldsWith64.F32, F32_DTYPE, 4),
        #         (FieldsWith64.F64, F64_DTYPE, 4),
        #         (FieldsWith64.U32, U32_DTYPE, 6),
        #         (FieldsWith64.I32, I32_DTYPE, 1),
        # )
        raise NotImplementedError("Support for a shared memory word size of 8")
    else:
        struct_field_descriptions = (
                (Fields.F32, F32_DTYPE, 4),
                (Fields.U32, U32_DTYPE, 6),
                (Fields.I32, I32_DTYPE, 1),
        )

    struct_fields.add_fields(struct_field_descriptions)

    print(f"struct_fields.total_size_per_thread = {struct_fields.struct_size()}")
    print(f"struct_fields.total_size_for_all_threads() = {struct_fields.struct_size_for_all_threads(number_threads_per_block)}")
    print(f"offsets = {struct_fields.field_info()}")

    struct_info = (struct_fields.struct_size(), cuda.to_device(np.array(struct_fields.field_info())))
    print(f"Total shared memory size: {struct_fields.struct_size_for_all_threads(number_threads_per_block)}")
    # Launch the kernel
    kernel_parameters = (struct_info, )
    # launch_parameters: [grid_dim, block_dim, stream, dyn_shared_mem_size]
    launch_parameters = (1, number_threads_per_block, 0, struct_fields.struct_size_for_all_threads(number_threads_per_block))
    run_with_small_arrays[launch_parameters](*kernel_parameters)

    cuda.synchronize()


def main():
    check_bank_allocation_for_small_arrays()
    check_bank_allocation_for_ligation_optimization()

    
if __name__ == '__main__':
    main()
