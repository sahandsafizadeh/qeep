package dispatch

import "github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"

func init() {
	gradtrack.RegisterFullFunc(Full)
	gradtrack.RegisterTransferFunc(Transfer)
}
