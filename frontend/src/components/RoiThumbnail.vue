<template>
  <div class="roi-thumb" :class="{ 'is-loaded': loaded }">
    <img
      :src="src"
      :alt="alt"
      loading="lazy"
      decoding="async"
      @load="onLoad"
      @error="onError"
    />

    <!--
      The overlay is an SVG in the image's own coordinate system, so the boxes
      scale with the thumbnail without any resize maths. preserveAspectRatio
      matches object-fit: contain on the <img>, which keeps the outlines on the
      objects when the tile is not the image's aspect ratio.
    -->
    <svg
      v-if="loaded && !failed && viewBox"
      class="roi-overlay"
      :viewBox="viewBox"
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      <rect
        v-for="(box, index) in drawn"
        :key="index"
        :x="box.x"
        :y="box.y"
        :width="box.w"
        :height="box.h"
        :stroke="box.colour"
        :stroke-width="strokeWidth"
        fill="none"
        vector-effect="non-scaling-stroke"
        rx="1"
      />
    </svg>

    <span v-if="failed" class="roi-failed">
      <slot name="failed">Image unavailable</slot>
    </span>

    <slot />
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  src: { type: String, required: true },
  alt: { type: String, default: '' },
  /** [[x, y, w, h, tag], ...] in the source image's pixel coordinates. */
  boxes: { type: Array, default: () => [] },
  width: { type: Number, default: null },
  height: { type: Number, default: null },
  /**
   * The project's class names, in order. Colours are assigned by position in
   * this list, which is the only way to guarantee they are far apart: hashing
   * the name put "0" and "1" within one degree of each other, and a
   * ten-digit project is exactly this tool's main use.
   */
  classes: { type: Array, default: () => [] }
})

const loaded = ref(false)
const failed = ref(false)
// Fallback for images whose dimensions were never cached.
const naturalWidth = ref(null)
const naturalHeight = ref(null)

const onLoad = (event) => {
  naturalWidth.value = event.target.naturalWidth
  naturalHeight.value = event.target.naturalHeight
  loaded.value = true
}

const onError = () => {
  failed.value = true
  loaded.value = true
}

const boxWidth = computed(() => props.width || naturalWidth.value)
const boxHeight = computed(() => props.height || naturalHeight.value)

const viewBox = computed(() =>
  boxWidth.value && boxHeight.value ? `0 0 ${boxWidth.value} ${boxHeight.value}` : null
)

/**
 * A stable colour per class.
 *
 * Position in the project's class list drives the hue, stepped by the golden
 * angle so consecutive classes land on opposite sides of the wheel and any
 * number of them stays maximally separated. Because the class list is sorted
 * server-side, a class keeps its colour across images and sessions.
 *
 * A tag not in the list (an annotation file edited by hand, say) falls back to
 * a hash with proper avalanche, so it is at least stable and distinct.
 */
const GOLDEN_ANGLE = 137.508

const fnv1a = (text) => {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return hash
}

const paletteIndex = computed(() => {
  const map = new Map()
  props.classes.forEach((name, index) => map.set(String(name), index))
  return map
})

const colourFor = (tag) => {
  const key = String(tag ?? '')
  const index = paletteIndex.value.get(key)
  const position = index === undefined ? fnv1a(key) : index
  const hue = (position * GOLDEN_ANGLE) % 360
  // Alternating lightness gives a second axis of separation, so even two
  // classes that land on a similar hue stay tellable apart.
  const lightness = position % 2 === 0 ? 64 : 72
  return `hsl(${hue.toFixed(1)} 90% ${lightness}%)`
}

const drawn = computed(() => {
  if (!boxWidth.value || !boxHeight.value) return []
  return props.boxes
    .map((box) => {
      const [x, y, w, h, tag] = box
      return { x, y, w, h, colour: colourFor(tag) }
    })
    .filter((b) => b.w > 0 && b.h > 0)
})

// A hairline at thumbnail size; non-scaling-stroke keeps it constant however
// far the viewBox is scaled down.
const strokeWidth = 1.75
</script>

<style scoped>
.roi-thumb {
  position: relative;
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: var(--bg);
}

.roi-thumb img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  opacity: 0;
  transition: opacity var(--t);
}

.roi-thumb.is-loaded img {
  opacity: 1;
}

.roi-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.roi-failed {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem;
  text-align: center;
  font-size: var(--fs-xs);
  color: var(--text-3);
}
</style>
