import queue
def calcFrequency(S):
    table=dict()
    for s in S:
        table[s]=table.get(s,0)+1
    return table


class Node:
    def __init__(self,symbol=''):
        self.char=symbol
        self.freq=0
        self.left=None
        self.right=None
    def __lt__(self, other):
        self.freq<other.freq

def AssignCodes(node, code, CodeTable):
    if node.char!='':
        CodeTable[node.char] = code
    else:
        AssignCodes(node.left, code + '0', CodeTable)
        AssignCodes(node.right, code + '1',CodeTable)

def Huffman(S):       # S: 입력 문자열 (문자열 길이 n)
    freqTable=calcFrequency(S)
    Q=queue.PriorityQueue()
    for s in freqTable:
        node=Node(s);node.freq=freqTable[s]
        Q.put(node)    # Q: 우선 순위 큐
    while Q.qsize() > 1:
        x = Q.get()  # 최소 빈도 노드
        y = Q.get()  # 두 번째 최소 빈도 노드
        z = Node('')
        z.left = x
        z.right = y
        z.freq = x.freq + y.freq
        Q.put(z)

    root = Q.get()  # 루트 노드
    codeTable={}
    AssignCodes(root, "",codeTable)

    return codeTable

S=input("압축할 문자열을 입력하세요.\n")

table=Huffman(S)
print("\n",table)

