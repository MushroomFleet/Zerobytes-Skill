# Zerobytes-Skill

**Position-is-Seed Procedural Generation for Claude Code Assistants**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude Skill](https://img.shields.io/badge/Claude-Skill-blueviolet)](https://github.com/MushroomFleet/Zerobytes-Skill)

> *"The coordinate IS the seed. Zero bytes store infinity."*

---

## What is Zerobytes?

Zerobytes is a Claude skill that teaches AI coding assistants the **position-is-seed paradigm** for procedural generation. Instead of sequential state iteration (like Braben's 1984 Tribonacci generator), Zerobytes uses pure coordinate hashing—enabling O(1) random access to any point in infinite space.

When you install this skill, Claude gains expertise in:

- **Infinite world generation** (Minecraft-style chunks, No Man's Sky planets)
- **Deterministic procedural content** (same seed = same world, always)
- **GPU-parallelizable generation** (thousands of elements simultaneously)
- **Spatially coherent noise** (biomes, regions, terrain that makes sense)
- **Hierarchical constraint systems** (galaxies → systems → planets → terrain)

---

## The Five Laws

Every procedural system Zerobytes designs satisfies these laws:

| Law | Requirement |
|-----|-------------|
| **O(1) Access** | Any position computable without iterating predecessors |
| **Parallelism** | Elements depend only on coordinates, never siblings |
| **Coherence** | Adjacent coordinates produce related values |
| **Hierarchy** | Child seeds derive from parent seeds + local position |
| **Determinism** | Same inputs → same outputs across all machines |

---

## Installation

### For Claude Desktop / Claude Code

1. Download `zerobytes.skill` from the [Releases](https://github.com/MushroomFleet/Zerobytes-Skill/releases) page
2. Add to your Claude skills directory
3. The skill auto-triggers on procedural generation requests

### For Claude Projects

Upload `zerobytes.skill` to your project's knowledge base.

### Manual Integration

Copy the contents of `SKILL.md` into your project's system prompt or CLAUDE.md file.

---

## Usage

Once installed, the skill triggers automatically when you ask Claude about:

- "Create an infinite procedural world"
- "Generate deterministic terrain from coordinates"
- "Build a seed-based dungeon generator"
- "Make a star system generator like Elite Dangerous"
- "Why are my procedural results different on different machines?"
- "How do I parallelize my world generation?"

### Example Prompts

```
"Create a Minecraft-style chunk generator where each chunk 
is deterministically generated from its coordinates"
```

```
"I'm getting different procedural results on different computers 
with the same seed. Debug my generation code."
```

```
"Design a galaxy generator with 400 billion star systems 
that I can query in any order"
```

---

## Core Pattern

The fundamental pattern Zerobytes teaches:

```python
import struct
import xxhash

def position_hash(x: int, y: int, z: int, salt: int = 0) -> int:
    """Hash coordinates directly. No iteration required."""
    h = xxhash.xxh64(seed=salt)
    h.update(struct.pack('<qqq', x, y, z))
    return h.intdigest()

# Jump to ANY position instantly
seed_at_origin = position_hash(0, 0, 0, world_seed)
seed_at_far_corner = position_hash(999999, 999999, 999999, world_seed)
# Both operations have identical cost: O(1)
```

Compare to traditional sequential generation:

```python
# OLD WAY: O(n) to reach position n
state = initial_seed
for i in range(1000000):  # Must iterate through all preceding positions
    state = next_state(state)
result = generate(state)

# ZEROBYTES WAY: O(1) for any position
result = generate(position_hash(1000000, 0, 0, seed))
```

---

## What's Included

```
Zerobytes-Skill/
├── README.md              # This file
├── zerobytes.skill        # Packaged skill for Claude
├── zerobytes/
│   └── SKILL.md           # Skill source
├── Zerobytes-Agent.md     # Full agentic system prompt (for CLAUDE.md)
├── ZeroBytes-mastery.md   # Comprehensive methodology guide
└── BeyondTribonacci-EndlessWorlds.md  # Technical deep-dive
```

### File Descriptions

| File | Purpose | Use When |
|------|---------|----------|
| `zerobytes.skill` | Compact skill package | Adding to Claude skills directory |
| `Zerobytes-Agent.md` | Full system prompt | Dedicated procedural generation projects |
| `ZeroBytes-mastery.md` | Teaching document | Training other AI assistants |
| `BeyondTribonacci-EndlessWorlds.md` | Technical reference | Deep implementation details |

---

## Comparison: Tribonacci vs Position-Hash

| Aspect | Tribonacci (1984) | Zerobytes (2025) |
|--------|-------------------|------------------|
| Access time | O(n) sequential | O(1) direct |
| Parallelization | Single-threaded only | Unlimited GPU cores |
| Spatial coherence | None (unrelated neighbors) | Noise-layered regions |
| Memory | 6 bytes state | 0 bytes (pure function) |
| Period | ~2^48 then cycles | Infinite (hash space) |
| Multi-scale | Not supported | Hierarchical inheritance |

---

## Quick Reference

### Hash Function Selection

| Platform | Recommended | Avoid |
|----------|-------------|-------|
| Python | `xxhash`, `blake2b` | `hash()`, `random` |
| C/C++ | xxHash3, MurmurHash3 | `rand()`, `srand()` |
| GPU/GLSL | PCG, Wang hash | LCG |
| JavaScript | `xxhash-wasm`, cyrb53 | `Math.random()` |

### Anti-Patterns to Avoid

```python
# ❌ Sequential iteration
for i in range(target): state = next(state)

# ❌ Platform-dependent hash
seed = hash((x, y))

# ❌ Time-based seeding
seed = int(time.time())

# ❌ Global RNG state
random.random()

# ❌ Neighbor dependency during generation
def gen(x, y): return f(gen(x+1, y))
```

### Correct Pattern

```python
# ✅ Direct coordinate hash
result = position_hash(x, y, z, world_seed)

# ✅ Coherent noise for regional properties
biome = coherent_noise(x * 0.01, y * 0.01, seed)

# ✅ Hierarchical seeds
planet_seed = position_hash(system_seed, orbit_index, 0, 0)
```

---

## Applications

Zerobytes methodology applies to:

- **Open-world games**: Infinite terrain, chunk-based loading
- **Space simulations**: Star systems, galaxies, planetary surfaces
- **Roguelikes**: Deterministic dungeon generation, seeded runs
- **Simulation games**: City generation, ecosystem modeling
- **Procedural art**: Texture synthesis, generative visuals
- **Testing**: Reproducible random test data generation

---

## Background

This skill distills techniques from David Braben's Elite (1984) and advances them for modern hardware. Where Braben's Tribonacci generator was optimal for 6502 processors with 32KB RAM, Zerobytes targets:

- Multi-core CPUs with SIMD instructions
- GPUs with thousands of parallel cores
- 64-bit address spaces
- Distributed computation across machines

The core insight remains Braben's: **generate, don't store**. But the implementation evolves from sequential state machines to pure coordinate functions.

---

## Contributing

Contributions welcome! Areas of interest:

- Additional language implementations (Rust, Go, C#)
- GPU compute shader examples
- Specialized generators (cities, ecosystems, music)
- Performance benchmarks
- Integration guides for game engines

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## 📚 Citation

### Academic Citation

If you use this codebase in your research or project, please cite:

```bibtex
@software{zerobytes_skill,
  title = {Zerobytes-Skill: Position-is-Seed Procedural Generation for AI Coding Assistants},
  author = {Drift Johnson},
  year = {2025},
  url = {https://github.com/MushroomFleet/Zerobytes-Skill},
  version = {1.0.0}
}
```

---

## Donate

If this skill has helped your project, consider supporting continued development:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Development-FF5E5B?logo=ko-fi)](https://ko-fi.com/mushroomfleet)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsor-EA4AAA?logo=github)](https://github.com/sponsors/MushroomFleet)

---

<p align="center">
  <i>Zero bytes store infinity. The universe springs complete from coordinates.</i>
</p>
