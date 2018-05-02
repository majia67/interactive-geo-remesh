#include "types.hpp"

Buffer::Buffer(int width, int height)
{
    R.resize(width, height);
    G.resize(width, height);
    B.resize(width, height);
    A.resize(width, height);
}

void Buffer::operator*=(const Buffer& b)
{
    R = R.array() * b.R.array();
    G = G.array() * b.G.array();
    B = B.array() * b.B.array();
    A = A.array() * b.A.array();
}
