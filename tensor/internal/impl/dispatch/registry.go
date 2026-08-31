package dispatch

import "github.com/sahandsafizadeh/qeep/tensor/internal/gradtrack"

func init() {
	gradtrack.RegisterTransferFunc(Transfer)
}
