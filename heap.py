import numpy as np


class Heap:
    def __init__(self):
        self.data = None
        self.heap = None
        self.heapPos = None
        self.size = 0
        self.heapItemCount = 0
        return

    def ClearHeap(self):
        self.heap = None
        self.heapPos = None
        self.heapItemCount = 0

    def SetDataPointer(self, items):
        self.data = items
        self.size = len(items)

    def Build(self):
        self.ClearHeap()
        self.heapItemCount = self.size
        self.heap = np.zeros(self.size + 1)
        self.heapPos = np.zeros(self.size)

        for i in range(self.heapItemCount):
            self.heapPos[i] = i + 1

        for i in range(1, self.heapItemCount + 1):
            self.heap[i] = i - 1

        if self.heapItemCount <= 1:
            return

        start = int(self.heapItemCount / 2)

        for ix in range(start, 0, -1):
            self.HeapMoveDown(ix)

    def IsSmaller(self, ix1, ix2):
        ix1 = int(ix1)
        ix2 = int(ix2)
        result = self.data[int(self.heap[ix1])] < self.data[int(self.heap[ix2])]
        return result

    def SwapItems(self, ix1, ix2):

        ix1 = int(ix1)
        ix2 = int(ix2)

        t = int(self.heap[ix1])
        self.heap[ix1] = self.heap[ix2]
        self.heap[ix2] = t

        self.heapPos[int(self.heap[ix1])] = ix1
        self.heapPos[int(self.heap[ix2])] = ix2

    def HeapMoveDown(self, ix):
        ix = int(ix)
        org = int(ix)
        child = int(ix * 2)

        while child + 1 <= self.heapItemCount:
            if self.IsSmaller(child, child + 1):
                child += 1
            if not self.IsSmaller(ix, child):
                return ix != org

            self.SwapItems(ix, child)
            ix = child
            child = ix * 2

        if child <= self.heapItemCount:
            if self.IsSmaller(ix, child):
                self.SwapItems(ix, child)
                return True

        return ix != org

    def MoveItemDown(self, id):
        id = int(id)
        return self.HeapMoveDown(self.heapPos[id])

    def GetTopItemID(self):
        return self.heap[1]

    def Pop(self):
        self.SwapItems(1, self.heapItemCount)
        self.heapItemCount -= 1
        self.HeapMoveDown(1)

    def GetIDFromHeap(self, heapIndex):
        heapIndex = int(heapIndex)
        return self.heap[heapIndex + 1]
