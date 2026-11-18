# Claude Design System
### Single Source of Truth — Reverse-Engineered from claude.ai & Cross-Referenced Sources

> **Status**: Community-derived, not officially published by Anthropic. Values confirmed against live CSS inspection, Anthropic's official Brand Guidelines skill, third-party palette databases (Mobbin, shadcn theme registry), and Geist Studio's case study on the original Anthropic.com brand.
>
> Last verified: April 2026.

---

## 1. Design Principles

Claude's visual identity is built on one core decision: warmth over convention. Every choice — colour, type, shadow, radius — tilts toward the human end of the spectrum. The palette contains no cool grays, no clinical whites, no blue-tinted surfaces. The result reads as a literary space rather than a software interface.

Three principles govern the system:

**Warm or nothing.** Every neutral must carry a yellow-brown undertone. If a value looks cool or blue, it's wrong.

**Serif for content, sans for function.** Headline copy uses the custom serif at a single weight. UI elements, labels, and body text use the sans. The distinction is absolute.

**Ring over drop.** Depth comes from 1px ring shadows and background-level contrast shifts, not from traditional drop shadows. Drop shadows appear only at very low opacity (≤ 0.05).

---

## 2. Colour Tokens

All values confirmed against live CSS and cross-referenced with Anthropic's official Brand Guidelines skill and the shadcn Claude theme.

### Core Surfaces

| Token | Hex | Role |
|---|---|---|
| `--color-bg-primary` | `#f5f4ed` | Page background — warm parchment. Never substitute `#ffffff`. |
| `--color-bg-elevated` | `#faf9f5` | Card and container surface on top of primary |
| `--color-bg-white` | `#ffffff` | Reserved for specific button surfaces only |
| `--color-bg-dark` | `#141413` | Dark section background — warm near-black with olive tint |
| `--color-bg-dark-elevated` | `#30302e` | Elevated containers on dark surfaces |

### Text

| Token | Hex | Role |
|---|---|---|
| `--color-text-primary` | `#141413` | Primary body text — same as dark background |
| `--color-text-secondary` | `#5e5d59` | Secondary body, de-emphasised content |
| `--color-text-tertiary` | `#87867f` | Captions, metadata, footnotes |
| `--color-text-button-muted` | `#4d4c48` | Button text on warm-sand surfaces |
| `--color-text-dark-primary` | `#faf9f5` | Primary text on dark surfaces |
| `--color-text-dark-secondary` | `#b0aea5` | Secondary text on dark surfaces |

### Brand & Accent

| Token | Hex | Role |
|---|---|---|
| `--color-brand` | `#c96442` | Primary CTA only — terracotta. Use sparingly. |
| `--color-accent` | `#d97757` | Text accents, links on dark surfaces |
| `--color-accent-blue` | `#6a9bcc` | Secondary brand accent (not interactive) |
| `--color-accent-green` | `#788c5d` | Secondary brand accent (not interactive) |
| `--color-focus` | `#3898ec` | Input focus rings only — the sole cool-toned value |
| `--color-error` | `#b53333` | Error states |

### Borders & Rings

| Token | Hex | Role |
|---|---|---|
| `--color-border-subtle` | `#f0eee6` | Default card borders on light surfaces |
| `--color-border-strong` | `#e8e6dc` | Dividers, prominent containment on light surfaces |
| `--color-border-dark` | `#30302e` | Borders on dark surfaces |
| `--color-ring-default` | `#d1cfc5` | Interactive element ring on hover/focus |
| `--color-ring-subtle` | `#dedc01` | Lighter interactive surfaces |
| `--color-ring-deep` | `#c2c0b6` | Active/pressed states |

### Interactive Surfaces

| Token | Hex | Role |
|---|---|---|
| `--color-surface-sand` | `#e8e6dc` | Warm-sand button background |

---

## 3. Typography

### Typefaces

The custom type family uses the internal naming below. For external use, the fallbacks are reliable substitutes.

| Role | Primary | Fallback |
|---|---|---|
| Headlines | Custom serif (`__copernicus_669e4a`) | `Georgia, ui-serif, serif` |
| UI / Body | Custom sans | `system-ui, Arial, sans-serif` |
| Code | Custom mono | `ui-monospace, monospace` |

> Note: The custom serif is a bespoke Copernicus-based typeface developed for Anthropic. It is not publicly available. Georgia is the closest accessible substitute.

### Scale

| Role | Size | Weight | Line Height | Notes |
|---|---|---|---|---|
| Display / Hero | 64px | 500 | 1.10 | Max size; scales down to ~25px on mobile |
| Section Heading | 52px | 500 | 1.20 | |
| Sub-heading Large | 36px | 500 | 1.30 | |
| Sub-heading | 32px | 500 | 1.10 | Card titles |
| Sub-heading Small | 25px | 500 | 1.20 | |
| Feature Title | 21px | 500 | 1.20 | |
| Body Serif | 17px | 400 | 1.60 | Editorial passages only |
| Body Large | 20px | 400 | 1.60 | Intro paragraphs |
| Body / Nav | 17px | 400–500 | 1.60 | Standard UI text |
| Body Standard | 16px | 400–500 | 1.50 | |
| Caption | 14px | 400 | 1.43 | Metadata |
| Label | 12px | 400–500 | 1.50 | Letter-spacing: 0.12px |
| Overline | 10px | 400 | 1.60 | Uppercase. Letter-spacing: 0.5px |
| Code | 15px | 400 | 1.60 | Letter-spacing: −0.32px |

**Rules:**
- All serif headings use weight 500 only. Never bold.
- Body line-height is 1.60 — generous by design. Do not reduce below 1.40.
- Label and overline text (≤12px) requires explicit letter-spacing for legibility.

---

## 4. Components

### Buttons

**Warm Sand** — secondary workhorse  
`background: #e8e6dc` · `color: #4d4c48` · `radius: 8px` · `padding: 0 12px 0 8px`  
`box-shadow: #e8e6dc 0 0 0 0, #d1cfc5 0 0 0 1px`

**Brand Terracotta** — primary CTA  
`background: #c96442` · `color: #faf9f5` · `radius: 8–12px`  
`box-shadow: #c96442 0 0 0 0, #c96442 0 0 0 1px`

**White Surface** — elevated light context  
`background: #ffffff` · `color: #141413` · `radius: 12px` · `padding: 8px 16px 8px 12px`

**Dark Charcoal** — inverted emphasis  
`background: #30302e` · `color: #faf9f5` · `radius: 8px` · `padding: 0 12px 0 8px`

**Dark Primary** — on dark-theme surfaces  
`background: #141413` · `color: #b0aea5` · `radius: 12px` · `padding: 9.6px 16.8px`  
`border: 1px solid #30302e`

### Cards

`background: #faf9f5` (light) or `#30302e` (dark)  
`border: 1px solid #f0eee6` (light) or `1px solid #30302e` (dark)  
`border-radius: 8px` (standard) · `16px` (featured) · `32px` (hero/media)  
`box-shadow: rgba(0,0,0,0.05) 0 4px 24px` (elevated only)

### Inputs & Form States

**Default**  
`color: #141413` · `padding: 1.6px 12px` · `border-radius: 12px`  
`border: 1px solid #e8e6dc` · `background: #faf9f5`

**Focus**  
`border-color: #3898ec` + `box-shadow: 0 0 0 3px rgba(56,152,236,0.15)`  
The only cool-toned moment in the system. Required for accessibility.

**Disabled**  
`background: #f0eee6` · `color: #87867f` · `border-color: #f0eee6` · `cursor: not-allowed`  
Opacity reduction alone is insufficient — colour shift signals the state.

**Error**  
`border-color: #b53333` + `box-shadow: 0 0 0 3px rgba(181,51,51,0.12)`  
Error message text in `#b53333` below the field.

**Read-only**  
`background: #f5f4ed` · `border-color: #f0eee6` · `color: #5e5d59` · `cursor: default`  
Visually distinct from disabled — content is legible and selectable.

**Success** (post-validation)  
No green border. Claude's palette doesn't use a success green. Confirmation is communicated through message copy, not input border colour.

### Navigation

Sticky. Warm background. Border: `1px solid #f0eee6` (light) or `1px solid #30302e` (dark).  
Links in `#141413`, `#5e5d59`, or `#3d3d3a`. CTA is Brand Terracotta or White Surface button.

---

## 5. Depth & Elevation

| Level | Treatment | Use |
|---|---|---|
| Flat | No shadow, no border | Page background, inline text |
| Contained | `1px solid` border (warm-toned) | Standard cards, sections |
| Ring | `0 0 0 1px` ring shadow | Interactive elements, hover states |
| Whisper | `rgba(0,0,0,0.05) 0 4px 24px` | Elevated cards, screenshots |
| Inset | `inset 0 0 0 1px` at 15% opacity | Pressed button states |

The most dramatic depth shift is the light/dark section alternation (`#f5f4ed` <-> `#141413`). This is the primary depth tool — entire sections change ambient light level to create reading rhythm.

---

## 6. Spacing & Layout

Base unit: **8px**  
Scale: 3 · 4 · 6 · 8 · 10 · 12 · 16 · 20 · 24 · 30px

Max container width: ~1200px, centred.  
Section vertical padding: 80–120px.  
Card internal padding: 24–32px.

### Border Radius Scale

| Size | Value | Typical use |
|---|---|---|
| Sharp | 4px | Minimal inline elements |
| Subtle | 6–8px | Standard buttons, cards |
| Generous | 12px | Primary buttons, inputs |
| Large | 16px | Featured containers, video |
| XL | 24px | Tags, highlighted containers |
| Max | 32px | Hero containers, large media |

### Responsive Breakpoints

| Name | Width | Key changes |
|---|---|---|
| Small mobile | <479px | Stacked, compact |
| Mobile | 479–640px | Single column, hamburger nav |
| Large mobile | 640–767px | Slightly wider |
| Tablet | 768–991px | 2-column grids, condensed nav |
| Desktop | 992px+ | Full layout, 64px hero type |

Hero text scales: 64px → 36px → ~25px.

---

## 7. Motion & Interaction

> **Caveat**: Claude.ai's exact transition values are not publicly documented. The guidance below is inferred from CSS inspection and aligned with established web motion principles. Treat as a best-practice specification, not a confirmed internal spec.

### Timing Scale

| Name | Duration | Use |
|---|---|---|
| Instant | 0ms | State changes that need no transition (disabled toggle) |
| Micro | 100–150ms | Hover colour shifts, button press |
| Standard | 200ms | Most UI transitions — border, background, shadow |
| Deliberate | 300ms | Panels opening, dropdown appearance |
| Expressive | 400–500ms | Page-level entrances, modal overlays |

### Easing

`ease-out` is the default for all entering elements — fast start, soft landing, feels responsive.  
`ease-in` for exiting elements — slow start, quick departure, reads as intentional removal.  
`ease-in-out` for elements moving between states (not entering or exiting) — tab switches, toggles.

Avoid `linear` for UI. Reserve it for progress bars and loading indicators where constant pace signals ongoing work.

### Property Priority

Prefer GPU-composited properties. In order of preference:

1. `opacity` and `transform` — compositor only, never trigger layout
2. `background-color`, `border-color`, `color`, `box-shadow` — paint only
3. Avoid transitioning `width`, `height`, `padding`, `margin` — these trigger layout reflow

### Interactive States (Buttons)

```
default  →  hover:  background lightens ~5%, ring shadow appears (150ms ease-out)
hover    →  active: scale(0.98), shadow reduces (100ms ease-in-out)  
active   →  release: returns to default (200ms ease-out)
any      →  focus-visible: focus ring appears (100ms ease-out)
```

### Reduced Motion

All transitions must respect `prefers-reduced-motion`. Replace transforms and fades with instant state switches when the user has requested reduced motion:

```css
@media (prefers-reduced-motion: reduce) {
  * { transition-duration: 0.01ms !important; }
}
```

---

## 8. Dark Mode

Claude.ai offers three appearance modes, confirmed via the official Help Centre: **Light**, **Dark**, and **Match System**. Match System follows the OS `prefers-color-scheme` preference. The toggle lives in Settings → Appearance.

### How it works

Dark mode is user-toggled (stored preference), not purely CSS media query driven. However, well-implemented recreations should support both mechanisms — manual toggle via a `data-theme` attribute on `<html>`, with `prefers-color-scheme` as the default when no preference is stored.

### Token Mapping: Light → Dark

| Light token | Light value | Dark equivalent | Dark value |
|---|---|---|---|
| `--color-bg-primary` | `#f5f4ed` | `--color-bg-dark` | `#141413` |
| `--color-bg-elevated` | `#faf9f5` | `--color-bg-dark-elevated` | `#30302e` |
| `--color-text-primary` | `#141413` | `--color-text-dark-primary` | `#faf9f5` |
| `--color-text-secondary` | `#5e5d59` | `--color-text-dark-secondary` | `#b0aea5` |
| `--color-border-subtle` | `#f0eee6` | `--color-border-dark` | `#30302e` |
| `--color-border-strong` | `#e8e6dc` | `--color-border-dark` | `#30302e` |

Brand terracotta (`#c96442`) and accent orange (`#d97757`) carry across both modes without adjustment — they are warm enough to read on both parchment and near-black.

The accent coral (`#d97757`) shifts roles in dark mode: on light surfaces it's a text accent; on dark surfaces it becomes the primary link and interactive text colour.

### Gradients in Dark Mode

The "gradient-free" description in the original document is an oversimplification. The claude.ai chat interface uses subtle gradients in specific places — most visibly as fade masks on scrollable containers and as soft ambient glows on the dark canvas. These are low-opacity, short-range gradients that create depth without visible colour bands. They are not decorative gradients in the traditional sense. The marketing site (claude.ai/home) is closer to gradient-free, relying on section alternation for depth.

---

## 9. Iconography

> **Caveat**: Claude.ai does not use a publicly named icon library. The following is observed from inspection.

### Style

- **Stroke-based**, not filled. All icons use outline strokes rather than solid fills.
- **Stroke weight**: approximately 1.5px at 20px size, scaling proportionally.
- **Corner treatment**: rounded joins and rounded caps — consistent with the system's soft radius language.
- **Size grid**: 16px (compact UI), 20px (standard), 24px (prominent actions).

### Colour

Icons inherit the text colour of their context. On light surfaces: `#141413` (primary) or `#5e5d59` (secondary). On dark surfaces: `#faf9f5` or `#b0aea5`. Icons never appear in terracotta — brand colour is reserved for buttons and text accents only.

### Behaviour

Interactive icons transition `color` and `opacity` on hover at standard timing (150–200ms ease-out). No scale transforms on icon hover — movement is colour-only.

---

## 10. Illustration System

Claude's illustrations are the most distinctive part of the visual identity. No other major AI product uses this approach.

### Style

- **Hand-drawn feeling**: vector art with slight organic irregularity — not pixel-perfect geometric shapes.
- **Conceptual, not literal**: illustrations represent ideas (connection, thought, scale) rather than depicting product UI or literal objects.
- **Figurative but abstract**: human-adjacent forms, not photorealistic.

### Colour Constraints

Illustrations use a strict three-colour palette:

| Colour | Hex | Role |
|---|---|---|
| Terracotta | `#c96442` | Primary illustration colour — dominant shapes |
| Near-Black | `#141413` | Line work, shadows, secondary shapes |
| Muted olive-green | `#788c5d` | Accent, tertiary elements |

White/negative space from the parchment background is treated as a fourth colour. Illustrations are never placed on dark backgrounds — they appear on parchment or ivory only.

### Usage

Used for conceptual or editorial sections, never for functional UI communication. Do not substitute illustrations with photographs or generic stock imagery. The hand-drawn quality is intentional — it contrasts with the precision of the UI components to create visual variety.

---

## 11. Z-Index & Layering

A layering contract prevents stacking context conflicts. Suggested scale:

| Layer | Z-index | Elements |
|---|---|---|
| Base | 0 | Page content, cards, sections |
| Sticky | 100 | Sticky navigation bar |
| Dropdown | 200 | Dropdown menus, select panels |
| Tooltip | 300 | Tooltips, hover popovers |
| Modal overlay | 400 | Modal backdrop |
| Modal content | 500 | Modal panels, drawers |
| Toast / notification | 600 | Toast messages, alerts |
| Maximum | 9999 | Debug overlays only |

> Note: Claude.ai's exact z-index values are not inspectable without authenticated access to the full application. The scale above follows standard design system conventions and is consistent with what can be observed.

---

## 12. Accessibility

### Contrast Ratios (Computed)

The following ratios were calculated using the WCAG 2.1 relative luminance formula. WCAG AA requires 4.5:1 for normal text, 3:1 for large text (18px+ regular or 14px+ bold).

| Pairing | Ratio | AA Normal | AA Large |
|---|---|---|---|
| Primary text `#141413` on parchment `#f5f4ed` | 16.7:1 | PASS | PASS |
| Secondary text `#5e5d59` on parchment | 5.98:1 | PASS | PASS |
| **Tertiary/stone `#87867f` on parchment** | **3.31:1** | **FAIL** | PASS |
| **Stone gray on ivory `#faf9f5`** | **3.47:1** | **FAIL** | PASS |
| Button text `#4d4c48` on warm-sand `#e8e6dc` | 6.87:1 | PASS | PASS |
| **Brand terracotta `#c96442` on ivory** | **3.70:1** | **FAIL** | PASS |
| **Accent orange `#d97757` on parchment** | **2.83:1** | **FAIL** | FAIL |
| Dark secondary `#b0aea5` on dark `#141413` | 8.29:1 | PASS | PASS |
| Ivory `#faf9f5` on dark `#141413` | 17.5:1 | PASS | PASS |

### Key findings

**Stone gray (`#87867f`) fails AA for normal-sized body text** on both parchment and ivory. It passes AA Large. Use it only for captions (14px+), metadata, and non-critical labels — never for paragraph text or anything a user needs to read at length.

**Terracotta (`#c96442`) and accent orange (`#d97757`) fail AA for text** on light surfaces. These colours should not be used for body text. Terracotta is safe for large headings (18px+) and buttons (where the contrast requirement applies to the button container, not text-on-background). Never use accent orange as standalone text colour on light surfaces.

### Focus Indicators

The focus ring uses `#3898ec` — the only cool-toned value in the system. This is intentional: the contrast with the warm palette makes focus states immediately visible. All interactive elements must show a visible focus ring on keyboard navigation. Do not suppress `:focus-visible` outlines.

Minimum focus ring: `0 0 0 3px rgba(56,152,236,0.4)` or `0 0 0 2px #3898ec` offset by 2px.

### Reduced Motion

See Section 7. All transitions must respect `prefers-reduced-motion: reduce`.

### Screen Reader Considerations

Icon-only buttons require `aria-label`. Decorative illustrations use `aria-hidden="true"`. The light/dark section alternation is purely visual — no semantic structure changes between sections.

---

## 13. Rules at a Glance

**Always:**
- Use `#f5f4ed` as the light page background, not white
- Use serif weight 500 for all headlines — no exceptions
- Keep all neutrals warm-toned (yellow-brown undertone)
- Use ring shadows (`0 0 0 1px`) for interactive states
- Set body line-height at 1.60
- Alternate light/dark sections for page rhythm

**Never:**
- Use cool blue-grays anywhere except `#3898ec` for focus states
- Use serif weight above 500
- Use drop shadows heavier than `rgba(0,0,0,0.05)`
- Use `#ffffff` as a page background
- Use border-radius below 6px on buttons or cards
- Use the monospace font for non-code content

---

## 14. What Isn't Known

This document is reverse-engineered. The following cannot be confirmed without access to Anthropic's internal design files:

- Exact names of the custom typefaces (the serif appears to be a bespoke Copernicus variant)
- Whether "Anthropic Serif / Sans / Mono" are the official internal names
- Precise transition timing and easing values (Section 7 is best-practice inference)
- Exact z-index values used in the live application (Section 11 follows convention)
- The specific icon library or whether icons are custom-drawn
- Official token naming conventions used internally
- Any design changes made after April 2026

For the most accurate implementation, inspect the live CSS at claude.ai directly.
