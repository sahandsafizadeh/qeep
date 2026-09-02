package persist

import "encoding/binary"

// FileExtension is the extension of the archive file holding a saved tensor.
const FileExtension = ".qeep"

const (
	dataFileName = "data"
	metaFileName = "meta"
)

var byteOrder = binary.LittleEndian
