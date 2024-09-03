"""
Copyright (c) 2024 Ian Cavén
MIT License

Shared memory structure definition calculator for dynamic shared memory in CUDA kernels.
Simplifies the calculation of the start of a member of a struct for each thread.

Shared memory structures are sequences of arrays of elements, each element being its own itemsize bytes large.
The arrays of elements may be padded at the end for alignment of subsequent arrays.  The full structure may be
padded to make the entire structure a multiple of the largest itemsize in bytes, since structures for each thread
follow, and so it must be possible for an element in a subsequent structure to be aligned on the same itemsize number
of bytes.

The shared memory structure must start at a different value (address % NUMBER_SHARED_MEMORY_BANKS) for each thread.
Constraints:
 - The offset of the start of the structure for a thread index modulo NUMBER_SHARED_MEMORY_BANKS must be:
        - a unique residue of the number of shared memory banks;  and
        - be a multiple of the shared memory word size; and
        - a multiple of the maximum itemsize.
   For the first requirement, (Address_in_bytes_of_a_structure // SHARED_MEMORY_WORD_SIZE) % NUMBER_SHARED_MEMORY_BANKS
   must be a unique number in the range(0, NUMBER_SHARED_MEMORY_BANKS), for groups of NUMBER_SHARED_MEMORY_BANKS number of threads.
   
 - The offset for each field of the structure must be an integral number of itemsize elements from the beginning of
   shared memory.  This is partially fulfilled by the structure being padded to make it an integral number of the maximum
   itemsize.  An additional requirement is to pad the space between structures so that it is also an integral number
   of the maximum itemsize.

** Note that the shared memory word size must be at least as large as the largest itemsize. **

"""

import math

import numba
import numpy as np
import numba.cuda as cuda
from numba.core import types
from numba.np import numpy_support

# These are the defaults
SHARED_MEMORY_WORD_SIZE = 4  # number of bytes in word in a shared memory bank
NUMBER_SHARED_MEMORY_BANKS = 32


# Shared memory slice helpers
@cuda.jit(device=True)
def slice_of_shared_memory(field_index, struct_info, thread_index):
    field_index = numba.int32(field_index)
    # These are the members of the tuple, but using them in the formulae below really slows down execution:
    #     itemsize, offset_in_items, plurality = struct_info[1][numba.int32(field_index)][:]
    first_index = (thread_index * struct_info[0] + struct_info[1][field_index][0] - 1) // \
                  struct_info[1][field_index][0] + struct_info[1][field_index][1]
    return slice(first_index, first_index + struct_info[1][field_index][2])


@cuda.jit(device=True)
def slice_of_shared_memory_across_threads(field_index, struct_info, number_threads):
    # Only works for 1D fields
    field_index = numba.int32(field_index)
    first_index = struct_info[1][field_index][1]
    return slice(first_index, first_index + (struct_info[0] * number_threads) // struct_info[1][field_index][0],
                 struct_info[0] // struct_info[1][field_index][0])


class SharedMemoryStructCalculator:
    
    def __init__(self, shared_memory_word_size=SHARED_MEMORY_WORD_SIZE,
                 number_shared_memory_banks=NUMBER_SHARED_MEMORY_BANKS):
        self.fields = {}  # Keyed by name: (value_type, plurality)
        self.max_itemsize = 0
        self.size_in_bytes = 0
        self.shared_memory_word_size = shared_memory_word_size
        self.number_shared_memory_banks = number_shared_memory_banks

    def add_fields(self, descriptions):
        for desc in descriptions:
            self.append_field(*desc)
                    
    def field_info(self):
        """
        Returns a sequence of information for each field in the structure (itemsize, offset in items, array size)
        that matches the sequence of fields appended.
        :return: A list of information appropriate for each field
        """
        offsets = []
        offset_in_bytes = 0
        for field_name, (dtype, plurality) in self.fields.items():
            offset_in_items = int(math.ceil(offset_in_bytes / dtype.itemsize))
            offsets.append((dtype.itemsize, offset_in_items, plurality))
            offset_in_bytes = (offset_in_items + plurality) * dtype.itemsize
        
        return offsets
    
    def append_field(self, name, dtype, plurality):
        assert isinstance(name, int)
        # The dtype is either a numba.core.types.Type or a numpy dtype
        # When the simulator is being used, the conversion of a numpy dtype must be done so that the itemsize attr is
        # available
        assert isinstance(dtype, (types.Type, np.dtype))
        if isinstance(dtype, types.Type):
            dtype = numpy_support.as_dtype(dtype)
        assert hasattr(dtype, 'itemsize')
        assert dtype.itemsize <= self.shared_memory_word_size, f'dtype itemsize {dtype.itemsize} must be <= the shared memory word size'
        # assert isinstance(plurality, int)
        self.fields[name] = (dtype, plurality)
        self.max_itemsize = max(self.max_itemsize, dtype.itemsize)
        self.size_in_bytes = (int(math.ceil(self.size_in_bytes / dtype.itemsize)) + plurality) * dtype.itemsize
    
    def struct_size(self):
        size_padded_for_maxsize_alignment = ((self.size_in_bytes + self.max_itemsize - 1) // self.max_itemsize) * self.max_itemsize
        size_adjustment_for_bank_access = (((size_padded_for_maxsize_alignment // self.shared_memory_word_size) %
                                            self.number_shared_memory_banks) + 1) * self.shared_memory_word_size
        # print(f"size_adjusted_for_bank_access: {size_adjustment_for_bank_access}, max item size: {self.max_itemsize}, "
        #       f"size_padded_for_maxsize_alignment: {size_padded_for_maxsize_alignment}")
        return size_padded_for_maxsize_alignment + size_adjustment_for_bank_access
    
    def struct_size_for_all_threads(self, number_threads_in_block) -> int:
        """
        Compute the total shared memory size in bytes.
        :param number_threads_in_block:     The total number of threads in the block
        :return: The size in bytes
        """
        return self.struct_size() * number_threads_in_block
