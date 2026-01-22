# Zero Bytes Mastery: Position-is-Seed for Claude Code Assistants

## Purpose

This document teaches Claude Code assistants how to help developers implement the **position-is-seed paradigm** for procedural generation. When a developer asks for help with infinite worlds, procedural content, or deterministic generation, use these patterns.

---

## The Core Insight in One Sentence

**The coordinate IS the seed.** Don't iterate to a location—hash the location directly.

```python
# OLD WAY (Tribonacci-style): Must iterate to reach position 1000
state = initial_seed
for i in range(1000):
    state = next_state(state)
result = generate(state)

# NEW WAY (Position-is-seed): Jump directly to any position
result = generate(hash(position))
```

---

## The Five Demands and How to Meet Them

### 1. O(1) Random Access

**The demand:** Any coordinate, instantly. No iteration.

**The pattern:** Hash the coordinates.

```python
import hashlib
import struct

def position_hash(x: int, y: int, z: int, salt: int = 0) -> int:
    """Hash 3D coordinates to a deterministic 64-bit integer."""
    data = struct.pack('<qqqQ', x, y, z, salt)
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return struct.unpack('<Q', digest)[0]

# Usage: Jump to ANY point instantly
seed_at_origin = position_hash(0, 0, 0)
seed_at_far_corner = position_hash(999999, 999999, 999999)  # Same cost!
```

**When a developer asks:** "How do I generate content at arbitrary positions without loading everything in between?"

**You answer:** Hash the position. The hash IS the seed. Show them this pattern.

**Production-grade alternative (faster):**

```python
# pip install xxhash
import xxhash

def fast_position_hash(x: int, y: int, z: int, salt: int = 0) -> int:
    h = xxhash.xxh64(seed=salt)
    h.update(struct.pack('<qqq', x, y, z))
    return h.intdigest()
```

---

### 2. Massive Parallelism

**The demand:** Generate thousands of things simultaneously on GPU.

**The pattern:** Each thread computes independently—no shared state.

```glsl
// GLSL Compute Shader: Generate 65,536 chunks in parallel
#version 450

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer ChunkBuffer {
    uint chunk_seeds[];
};

uniform ivec3 region_origin;
uniform uint world_salt;

// GPU-friendly hash (no branching, no memory access)
uint pcg_hash(uint input) {
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint position_to_seed(ivec3 pos) {
    // Combine coordinates into single hash input
    uint combined = uint(pos.x) ^ (uint(pos.y) << 10) ^ (uint(pos.z) << 20);
    return pcg_hash(combined ^ world_salt);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    // Convert linear index to 3D offset
    ivec3 offset = ivec3(idx & 0xF, (idx >> 4) & 0xF, (idx >> 8) & 0xF);
    ivec3 world_pos = region_origin + offset;
    
    // Each thread works independently—no synchronization needed
    chunk_seeds[idx] = position_to_seed(world_pos);
}
```

**When a developer asks:** "How do I parallelize my world generation?"

**You answer:** Make sure each unit of work (chunk, tile, star system) can be computed from ONLY its coordinates. No dependencies on neighbors during generation. Dependencies come AFTER generation, during combination.

**The critical rule:** If thread A needs thread B's result to compute its own, you've broken parallelism. Redesign so each thread needs only its coordinates.

---

### 3. Spatial Coherence

**The demand:** Neighbors should be related. A forest shouldn't randomly border a volcano.

**The pattern:** Layer coherent noise on top of position hashes.

```python
import numpy as np

def coherent_value(x: float, y: float, seed: int, octaves: int = 4) -> float:
    """
    Generate spatially coherent value at (x, y).
    Nearby points return similar values.
    """
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for i in range(octaves):
        # Sample noise at this frequency
        nx = x * frequency
        ny = y * frequency
        
        # Grid cell coordinates
        x0, y0 = int(np.floor(nx)), int(np.floor(ny))
        x1, y1 = x0 + 1, y0 + 1
        
        # Interpolation weights
        sx = nx - x0
        sy = ny - y0
        
        # Corner values from position hashes
        n00 = (position_hash(x0, y0, 0, seed + i) / 2**64) * 2 - 1
        n10 = (position_hash(x1, y0, 0, seed + i) / 2**64) * 2 - 1
        n01 = (position_hash(x0, y1, 0, seed + i) / 2**64) * 2 - 1
        n11 = (position_hash(x1, y1, 0, seed + i) / 2**64) * 2 - 1
        
        # Smoothstep interpolation
        sx = sx * sx * (3 - 2 * sx)
        sy = sy * sy * (3 - 2 * sy)
        
        # Bilinear interpolation
        nx0 = n00 * (1 - sx) + n10 * sx
        nx1 = n01 * (1 - sx) + n11 * sx
        n = nx0 * (1 - sy) + nx1 * sy
        
        value += amplitude * n
        max_value += amplitude
        amplitude *= 0.5  # Persistence
        frequency *= 2.0  # Lacunarity
    
    return value / max_value  # Normalize to [-1, 1]


def biome_at(x: float, y: float, seed: int) -> str:
    """Determine biome using multiple coherent noise layers."""
    temperature = coherent_value(x * 0.01, y * 0.01, seed)
    moisture = coherent_value(x * 0.01, y * 0.01, seed + 1000)
    
    # Biome selection based on coherent values
    if temperature > 0.3:
        return "desert" if moisture < 0 else "jungle"
    elif temperature > -0.3:
        return "plains" if moisture < 0 else "forest"
    else:
        return "tundra" if moisture < 0 else "taiga"
```

**When a developer asks:** "My random generation looks too chaotic/noisy. How do I make regions?"

**You answer:** Raw hashes produce white noise. Layer coherent noise (Perlin, Simplex, or value noise) to create smooth gradients. Use different noise layers for different properties (temperature, moisture, elevation), then combine them to determine the final result.

**The visual difference:**

```
White noise (raw hash):     Coherent noise (layered):
██░░██░░░███░░██           ██████████░░░░░░
░░██░░████░░██░░           ████████░░░░░░░░
██░░██░░██░░░░██           ██████░░░░░░░░░░
░░░░██░░██████░░           ████░░░░░░░░░░░░
                           ██░░░░░░░░░░░░░░
```

---

### 4. Multi-Scale Consistency

**The demand:** Zoom from galaxy view to ground detail without discontinuity.

**The pattern:** Hierarchical generation with parent-to-child constraint passing.

```python
from dataclasses import dataclass
from typing import List

@dataclass
class GalaxyRegion:
    x: int
    y: int
    seed: int
    stellar_density: float  # Derived from position in galaxy
    average_metallicity: float

@dataclass
class StarSystem:
    region: GalaxyRegion
    local_x: int
    local_y: int
    seed: int
    star_mass: float
    metallicity: float  # Constrained by region

@dataclass
class Planet:
    system: StarSystem
    orbit_index: int
    seed: int
    mass: float
    has_atmosphere: bool  # Constrained by star properties

def generate_region(gx: int, gy: int, galaxy_seed: int) -> GalaxyRegion:
    """Top level: Galaxy region from galactic coordinates."""
    seed = position_hash(gx, gy, 0, galaxy_seed)
    
    # Use coherent noise for galaxy-scale structure
    density = 0.5 + 0.5 * coherent_value(gx * 0.1, gy * 0.1, galaxy_seed)
    metallicity = 0.02 * (1.0 - abs(coherent_value(gx * 0.05, gy * 0.05, galaxy_seed + 1)))
    
    return GalaxyRegion(gx, gy, seed, density, metallicity)

def generate_system(region: GalaxyRegion, lx: int, ly: int) -> StarSystem:
    """Mid level: Star system constrained by region."""
    seed = position_hash(lx, ly, 0, region.seed)
    rng = hash_to_floats(seed)
    
    # Star mass: use region's density to influence probability of massive stars
    base_mass = 0.1 + rng[0] * 2.0
    mass_boost = region.stellar_density * rng[1]
    star_mass = base_mass * (1 + mass_boost)
    
    # Metallicity: inherit from region with local variation
    metallicity = region.average_metallicity * (0.5 + rng[2])
    
    return StarSystem(region, lx, ly, seed, star_mass, metallicity)

def generate_planet(system: StarSystem, orbit_index: int) -> Planet:
    """Bottom level: Planet constrained by star."""
    seed = position_hash(orbit_index, 0, 0, system.seed)
    rng = hash_to_floats(seed)
    
    # Mass depends on position and available material (metallicity)
    base_mass = rng[0] * 10  # Earth masses
    mass = base_mass * (1 + system.metallicity * 10)
    
    # Atmosphere retention depends on planet mass and star's radiation
    escape_velocity = mass ** 0.5
    stellar_radiation = system.star_mass ** 3.5
    has_atmosphere = escape_velocity > stellar_radiation * 0.1 * rng[1]
    
    return Planet(system, orbit_index, seed, mass, has_atmosphere)

def hash_to_floats(seed: int, count: int = 10) -> List[float]:
    """Extract multiple float values from a seed."""
    result = []
    current = seed
    for _ in range(count):
        current = position_hash(current, 0, 0, 0)
        result.append((current & 0xFFFFFFFF) / 0xFFFFFFFF)
    return result
```

**When a developer asks:** "How do I make sure my detail view matches my overview?"

**You answer:** Generate top-down. Parent objects pass constraints to children. The planet doesn't randomly decide its properties—it inherits constraints from its star, which inherited from its region. Each level's seed derives from its parent's seed plus its local position.

**The hierarchy pattern:**

```
Galaxy Seed ──┬── Region(0,0) ──┬── System(0,0) ──┬── Planet 0
              │                 │                 ├── Planet 1
              │                 │                 └── Planet 2
              │                 ├── System(0,1) ──┬── Planet 0
              │                 │                 └── Planet 1
              │                 └── System(1,0) ──── ...
              │
              ├── Region(0,1) ──── ...
              └── Region(1,0) ──── ...
```

Each node's seed = hash(parent_seed, local_position).

---

### 5. Deterministic Reproducibility

**The demand:** Same inputs produce same outputs. Always. Everywhere.

**The pattern:** Pure functions only. No global state. No time-based randomness.

```python
# BAD: Non-deterministic
import random
import time

def generate_bad(x, y):
    random.seed(time.time())  # Different every millisecond!
    return random.random()

# BAD: Hidden global state
_global_rng = random.Random()

def generate_also_bad(x, y):
    return _global_rng.random()  # Depends on call history!

# GOOD: Pure function of inputs
def generate_good(x: int, y: int, world_seed: int) -> float:
    """Pure function: same inputs ALWAYS produce same output."""
    seed = position_hash(x, y, 0, world_seed)
    return (seed & 0xFFFFFFFF) / 0xFFFFFFFF
```

**When a developer asks:** "Players are seeing different worlds with the same seed!"

**You check for:**

1. **Floating-point operations across platforms** - Use integer math for seeds
2. **Hash function differences** - Use a specific implementation, not platform default
3. **Implicit state** - Search for `random.random()` without explicit seed
4. **Order-dependent generation** - If generating A before B gives different results than B before A, there's hidden state

**The test for determinism:**

```python
def test_determinism():
    world_seed = 12345
    
    # Generate in order A
    result_a1 = generate_good(100, 200, world_seed)
    result_a2 = generate_good(300, 400, world_seed)
    result_a3 = generate_good(100, 200, world_seed)  # Same as a1?
    
    # Generate in order B (reversed)
    result_b2 = generate_good(300, 400, world_seed)
    result_b1 = generate_good(100, 200, world_seed)
    
    assert result_a1 == result_a3, "Same position must give same result"
    assert result_a1 == result_b1, "Order must not matter"
    assert result_a2 == result_b2, "Order must not matter"
    
    print("✓ Determinism verified")
```

---

## Common Developer Requests and Responses

### "I need Minecraft-style infinite terrain"

```python
@dataclass
class Chunk:
    cx: int  # Chunk coordinate, not block coordinate
    cz: int
    seed: int
    heightmap: np.ndarray  # 16x16 heights
    blocks: np.ndarray     # 16x256x16 block types

def generate_chunk(cx: int, cz: int, world_seed: int) -> Chunk:
    """Generate a 16x16 chunk at chunk coordinates (cx, cz)."""
    chunk_seed = position_hash(cx, cz, 0, world_seed)
    
    # Heightmap from coherent noise
    heightmap = np.zeros((16, 16), dtype=np.int32)
    for lx in range(16):
        for lz in range(16):
            # World-space coordinates for noise sampling
            wx = cx * 16 + lx
            wz = cz * 16 + lz
            
            # Multi-octave terrain height
            height = 64 + int(32 * coherent_value(wx * 0.01, wz * 0.01, world_seed))
            heightmap[lx, lz] = height
    
    # Block array (simplified)
    blocks = np.zeros((16, 256, 16), dtype=np.uint8)
    for lx in range(16):
        for lz in range(16):
            surface = heightmap[lx, lz]
            blocks[lx, :surface-3, lz] = 1   # Stone
            blocks[lx, surface-3:surface, lz] = 2  # Dirt
            blocks[lx, surface, lz] = 3      # Grass
    
    return Chunk(cx, cz, chunk_seed, heightmap, blocks)
```

### "I need No Man's Sky-style planet variety"

```python
def generate_planet_biosphere(planet_seed: int, parent_star_temp: float) -> dict:
    """Generate a planet's biosphere from seed and stellar constraints."""
    rng = hash_to_floats(planet_seed, 20)
    
    # Base temperature from star, modified by atmosphere
    atmosphere_thickness = rng[0]
    greenhouse_effect = atmosphere_thickness * rng[1] * 50
    base_temp = parent_star_temp + greenhouse_effect - 30
    
    # Water presence
    has_water = 0 < base_temp < 100 and rng[2] > 0.3
    
    # Flora density (requires water and suitable temp)
    flora_density = 0
    if has_water and -20 < base_temp < 60:
        flora_density = rng[3] * rng[4]  # Two rolls for rarity
    
    # Fauna complexity (requires flora)
    fauna_complexity = 0
    if flora_density > 0.2:
        fauna_complexity = int(rng[5] * 10)  # 0-10 scale
    
    # Dominant colors from seed (for visual distinctiveness)
    palette_seed = position_hash(planet_seed, 0, 0, 0)
    primary_hue = (palette_seed & 0xFF) / 255 * 360
    secondary_hue = ((palette_seed >> 8) & 0xFF) / 255 * 360
    
    return {
        "temperature": base_temp,
        "atmosphere": atmosphere_thickness,
        "has_water": has_water,
        "flora_density": flora_density,
        "fauna_complexity": fauna_complexity,
        "primary_hue": primary_hue,
        "secondary_hue": secondary_hue,
    }
```

### "I need roguelike dungeon generation that's reproducible"

```python
def generate_dungeon_room(room_x: int, room_y: int, floor: int, dungeon_seed: int) -> dict:
    """Generate a single room's contents deterministically."""
    room_seed = position_hash(room_x, room_y, floor, dungeon_seed)
    rng = hash_to_floats(room_seed, 15)
    
    # Room dimensions
    width = 5 + int(rng[0] * 10)
    height = 5 + int(rng[1] * 10)
    
    # Room type
    room_types = ["empty", "treasure", "monster", "trap", "shop", "shrine"]
    type_index = int(rng[2] * len(room_types))
    room_type = room_types[type_index]
    
    # Exits (deterministic based on position for connectivity)
    exits = {
        "north": position_hash(room_x, room_y - 1, floor, dungeon_seed) % 4 != 0,
        "south": position_hash(room_x, room_y + 1, floor, dungeon_seed) % 4 != 0,
        "east": position_hash(room_x + 1, room_y, floor, dungeon_seed) % 4 != 0,
        "west": position_hash(room_x - 1, room_y, floor, dungeon_seed) % 4 != 0,
    }
    
    # Contents based on room type and floor depth
    difficulty_scale = 1 + floor * 0.5
    
    contents = {
        "width": width,
        "height": height,
        "type": room_type,
        "exits": exits,
        "monster_count": int(rng[3] * 5 * difficulty_scale) if room_type == "monster" else 0,
        "treasure_value": int(rng[4] * 100 * difficulty_scale) if room_type == "treasure" else 0,
        "trap_damage": int(rng[5] * 20 * difficulty_scale) if room_type == "trap" else 0,
    }
    
    return contents
```

---

## Anti-Patterns to Catch

When reviewing developer code, flag these issues:

### Anti-Pattern 1: Sequential Iteration to Position

```python
# BAD: O(n) to reach position n
def get_value_bad(index, seed):
    state = seed
    for i in range(index):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
    return state

# GOOD: O(1) for any position
def get_value_good(index, seed):
    return position_hash(index, 0, 0, seed)
```

### Anti-Pattern 2: Shared Mutable State

```python
# BAD: Generator state shared between calls
class BadGenerator:
    def __init__(self, seed):
        self.state = seed
    
    def next(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

# GOOD: Pure function, no state
def good_generator(index: int, seed: int) -> int:
    return position_hash(index, 0, 0, seed)
```

### Anti-Pattern 3: Time-Based Seeds

```python
# BAD: Non-reproducible
seed = int(time.time())

# GOOD: Explicit seed
seed = 42  # Or from user input, or from parent object
```

### Anti-Pattern 4: Platform-Dependent Hashing

```python
# BAD: Python's hash() varies between runs and platforms
seed = hash((x, y))

# GOOD: Explicit hash implementation
seed = position_hash(x, y, 0, world_seed)
```

### Anti-Pattern 5: Generating Neighbors During Self-Generation

```python
# BAD: Circular dependency risk, breaks parallelism
def generate_cell_bad(x, y, seed):
    my_value = position_hash(x, y, 0, seed)
    neighbor_value = generate_cell_bad(x + 1, y, seed)  # Dependency!
    return (my_value + neighbor_value) / 2

# GOOD: Generate independently, combine later
def generate_cell_good(x, y, seed):
    return position_hash(x, y, 0, seed)

def smooth_cells(x, y, seed):
    """Separate pass for neighbor-dependent operations."""
    values = [generate_cell_good(x + dx, y + dy, seed) 
              for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    return sum(values) / len(values)
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                 POSITION-IS-SEED QUICK REFERENCE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CORE FORMULA:  result = process(hash(x, y, z, salt))          │
│                                                                 │
│  HASH CHOICE:                                                   │
│    • Python: hashlib.blake2b or xxhash                         │
│    • C/C++: xxHash3, MurmurHash3                               │
│    • GPU: PCG, Wang hash                                        │
│    • JavaScript: cyrb53, xxhash-wasm                           │
│                                                                 │
│  NOISE LAYERING:                                                │
│    Base layer:    coherent_noise(x * 0.001, ...)  (continents) │
│    Mid layer:     coherent_noise(x * 0.01, ...)   (regions)    │
│    Detail layer:  coherent_noise(x * 0.1, ...)    (local)      │
│    Micro layer:   hash(x, y) / MAX                (per-tile)   │
│                                                                 │
│  HIERARCHY PATTERN:                                             │
│    parent_seed = hash(grandparent_seed, parent_position)       │
│    child_seed = hash(parent_seed, child_local_position)        │
│                                                                 │
│  DETERMINISM CHECKLIST:                                         │
│    □ No time-based seeds                                        │
│    □ No global random state                                     │
│    □ No platform-dependent hash()                               │
│    □ No order-dependent generation                              │
│    □ Integer math for seeds (no float accumulation)            │
│                                                                 │
│  PARALLELISM RULE:                                              │
│    If generation of A requires result of B → redesign          │
│    Generate all independently, combine in separate pass        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Final Guidance for Claude Assistants

When helping developers with procedural generation:

1. **Always ask:** "What's your seed, and how does it flow to child objects?"

2. **Default to position hashing** unless there's a specific reason for sequential generation (there rarely is).

3. **Suggest coherent noise** when output looks "too random" or lacks regional consistency.

4. **Check for hidden state** when developers report inconsistent results across machines or sessions.

5. **Recommend hierarchical design** when developers struggle with consistency between zoom levels.

6. **Verify determinism** by generating the same position twice in different orders—results must match.

The universe springs complete from coordinates. Help developers build systems where any point can be queried instantly, independently, and reproducibly. Zero bytes store infinity.
