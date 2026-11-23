# Third-party notices

The opt-in CUTLASS Xe2 expert path (`INK2_CUTLASS_MOE=1`) vendors two files from [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) at commit `f10f6674499e6fdbee0a75b5624cd0cfe8041ce6`:

- `csrc/nvfp4_xe2_ext.cpp` (from `patches/inkling2-mini-vllm-xpu/nvfp4_xe2_ext.cpp`)

- `src/vllm_inkling2_xpu/ops/moe_cutlass.py` (from `patches/inkling2-mini-vllm-xpu/moe_cutlass.py`)

They are used under the following license:

```
MIT License

Copyright (c) 2026 SergiioB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
