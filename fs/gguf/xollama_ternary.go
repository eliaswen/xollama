package gguf

// The group-128 ternary types of PrismML's Bonsai 2, which opencoti runs and
// upstream's tables do not know. Without them a tensor's size reads as 0 and
// `create` refuses the file as "size overflows".
//
// The ids are ggml's (GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0) and llama.cpp's
// (LLAMA_FTYPE_MOSTLY_PQ2_0, its pre-rename value still in published GGUFs,
// and LLAMA_FTYPE_MOSTLY_PTQ1_0).
const (
	tensorTypePQ2_0  TensorType = 142
	tensorTypePTQ1_0 TensorType = 143

	fileTypePQ2_0       FileType = 141
	fileTypePQ2_0Legacy FileType = 142
	fileTypePTQ1_0      FileType = 143
)

// ternaryGroup is how many weights share one f16 scale.
const ternaryGroup = 128

// ternaryTypeSize is the bytes of one group: the f16 scale, then 2 bits a
// weight (PQ2_0) or five trits to the byte (PTQ1_0). 0 for any other type,
// which is upstream's answer for a type it cannot size.
func ternaryTypeSize(tt TensorType) int64 {
	switch tt {
	case tensorTypePQ2_0:
		return 2 + ternaryGroup/4
	case tensorTypePTQ1_0:
		return 2 + (ternaryGroup+4)/5
	default:
		return 0
	}
}

// ternaryBlockSize is the group for a ternary type, and upstream's 256 for
// any other type its own table does not list.
func ternaryBlockSize(tt TensorType) int64 {
	switch tt {
	case tensorTypePQ2_0, tensorTypePTQ1_0:
		return ternaryGroup
	default:
		return 256
	}
}

func ternaryTensorName(tt TensorType) string {
	switch tt {
	case tensorTypePQ2_0:
		return "pq2_0"
	case tensorTypePTQ1_0:
		return "ptq1_0"
	default:
		return "unknown"
	}
}

func ternaryFileTypeName(t FileType) string {
	switch t {
	case fileTypePQ2_0, fileTypePQ2_0Legacy:
		return "PQ2_0"
	case fileTypePTQ1_0:
		return "PTQ1_0"
	default:
		return "unknown"
	}
}
