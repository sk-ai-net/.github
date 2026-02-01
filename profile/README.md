# 📦 Repository Move Notice

## FED-SKaiNET

Federated-learning capability are implemented under [FED-SKaiNET](https://github.com/FED-SKaiNET)

# What SKaiNET Enables for Federated Learning

## Cross-Platform Tensor Math — One Codebase, Every Device

SKaiNET's `ExecutionContext` and `TensorOps` provide a single API for tensor operations (`add`, `subtract`, `mulScalar`, `divide`, `softmax`, `sigmoid`, etc.) that works identically across JVM, Android, iOS, JS, WASM, and native Linux/macOS. This means the entire FedAvg aggregation, gradient accumulation, and optimizer logic is written once in `commonMain` and runs everywhere.

## Typed Model Parameter Access

SKaiNET's `Module<FP32, Float>` and `ModuleNode` hierarchy lets federated code extract, iterate, and update named model parameters recursively. This is the foundation for:

- **`ParameterManager`** — extracting global weights from a model
- **Weight delta computation** — `ops.subtract(current, initial)` per parameter after local training
- **Applying aggregated updates** — writing server-aggregated weights back into client modules

## Weighted Averaging on Real Tensors

FedAvg's core operation — weighted average of client weight updates — is built directly on `ops.mulScalar()` and `ops.add()`. Without SKaiNET providing element-wise tensor math, this would require raw array manipulation per platform.

## Forward Pass and Loss Computation On-Device

`module.forward(input, ctx)` combined with `FederatedLoss` implementations (MSE, CrossEntropy, BinaryCE) using SKaiNET ops (`softmax`, `sigmoid`, `mean`, `sum`) makes local on-device training possible. Clients can train, compute loss, and return results — all through SKaiNET's module system.

## Optimizers Built on Tensor Primitives

`FederatedSGD` and `FederatedAdam` are implemented entirely with SKaiNET ops — momentum updates, squared gradient tracking, bias correction, weight decay — all via `ops.add`, `ops.multiply`, `ops.mulScalar`, `ops.divide`. No custom math kernels needed.

## Memory-Efficient Training on Edge Devices

SKaiNET's `ExecutionContext` enables:

- **`TensorPool`** — reuses tensor allocations via `ctx.zeros()` and `ctx.full()` to reduce GC pressure on mobile
- **`CheckpointingGradientTape`** — wraps the execution context to trade compute for memory during backward passes
- **`GradientAccumulator`** — accumulates gradients across micro-batches using `ops.add` and `ops.mulScalar`

## Data Loading Integration

The `DataLoader<T, V>` / `DataBatch<T, V>` abstraction is generic over SKaiNET's `DType` system, connecting to `skainet-data-api` and `skainet-data-simple` for batched tensor I/O on each platform.

---

**In short:** SKaiNET provides the typed, cross-platform tensor runtime and module system that lets federated learning — aggregation, local training, optimization, and memory management — be written as pure Kotlin multiplatform code without platform-specific math or model handling.


## sk-ai-net → **SKaiNET-developers**

The **sk-ai-net** GitHub organization has been **moved to `SKaiNET-developers`** to better reflect and serve its target audience: developers, contributors, and integrators working with SKaiNET projects.

✅ **Nothing is lost**  
All repositories, files, commit history, issues, and tags remain intact.

---

## 🔗 What You Need to Do

If you are using this repository locally, **please update your Git remote URL** to point to the new organization.

### 1️⃣ Check your current remote
```bash
git remote -v
```

### 2️⃣ Update the remote URL
Replace `sk-ai-net` with `SKaiNET-developers`:

```bash
git remote set-url origin https://github.com/SKaiNET-developers/<repository-name>.git
```

For SSH users:
```bash
git remote set-url origin git@github.com:SKaiNET-developers/<repository-name>.git
```

### 3️⃣ Verify the change
```bash
git remote -v
```

---

## 🌐 Update Your Links

Please update any references in documentation, CI/CD pipelines, submodules, scripts, and bookmarks.

Old links under:

```
github.com/sk-ai-net/...
```

should now use:

```
github.com/SKaiNET-developers/...
```

---

## 💬 Questions?

If you encounter any issues or broken links, please open an issue in the relevant repository under **SKaiNET-developers**.

Thanks for your support and happy hacking 🚀
