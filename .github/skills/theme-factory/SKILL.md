---
name: theme-factory
description: Toolkit for styling artifacts with a theme. These artifacts can be slides, docs, reportings, HTML landing pages, etc. There are 10 pre-set themes with colors/fonts that you can apply to any artifact that has been creating, or can generate a new theme on-the-fly.
license: Complete terms in LICENSE.txt
---


# Theme Factory Skill

This skill provides a curated collection of professional font and color themes themes, each with carefully selected color palettes and font pairings. Once a theme is chosen, it can be applied to any artifact.

## Purpose

To apply consistent, professional styling to presentation slide decks, use this skill. Each theme includes:
- A cohesive color palette with hex codes
- Complementary font pairings for headers and body text
- A distinct visual identity suitable for different contexts and audiences

## Usage Instructions

To apply styling to a slide deck or other artifact:

1. **Show the theme showcase**: Display the `theme-showcase.pdf` file to allow users to see the 10 general-purpose themes visually. Do not make any modifications to it; simply show the file for viewing.
2. **Check for Philips-branded work**: If the artifact needs to follow Philips branding, also display `philips-theme-showcase.pdf`. These Philips themes are derived from the Philips brand assets skill at `../../philips/philips-assets/` (which has its own SKILL.md with logo selection, font usage, and music cue guidance).
3. **Ask for their choice**: Ask which theme to apply to the deck
4. **Wait for selection**: Get explicit confirmation about the chosen theme
5. **Apply the theme**: Once a theme has been chosen, apply the selected theme's colors and fonts to the deck/artifact

## Themes Available

The following 13 themes are available. The first 10 are showcased in `theme-showcase.pdf`, and the Philips themes are showcased in `philips-theme-showcase.pdf`:

1. **Ocean Depths** - Professional and calming maritime theme
2. **Sunset Boulevard** - Warm and vibrant sunset colors
3. **Forest Canopy** - Natural and grounded earth tones
4. **Modern Minimalist** - Clean and contemporary grayscale
5. **Golden Hour** - Rich and warm autumnal palette
6. **Arctic Frost** - Cool and crisp winter-inspired theme
7. **Desert Rose** - Soft and sophisticated dusty tones
8. **Tech Innovation** - Bold and modern tech aesthetic
9. **Botanical Garden** - Fresh and organic garden colors
10. **Midnight Galaxy** - Dramatic and cosmic deep tones
11. **Philips PowerPoint** - Core Philips presentation system with brand blue, white, and light gray layouts
12. **Philips Webinar Light** - Bright Philips gradient theme for polished webinars and external-facing sessions
13. **Philips Webinar Dark** - Deep Philips gradient theme for cinematic webinar intros, outros, and title cards

## Theme Details

Each theme is defined in the `themes/` directory with complete specifications including:
- Cohesive color palette with hex codes
- Complementary font pairings for headers and body text
- Distinct visual identity suitable for different contexts and audiences

The Philips themes additionally align to the bundled Philips brand assets, including Neue Frutiger World and the Philips blue gradient system.

The Philips-specific showcase PDF is generated from `generate-philips-showcase.py`.

## Application Process

After a preferred theme is selected:
1. Read the corresponding theme file from the `themes/` directory
2. Apply the specified colors and fonts consistently throughout the deck
3. Ensure proper contrast and readability
4. Maintain the theme's visual identity across all slides

## Create your Own Theme
To handle cases where none of the existing themes work for an artifact, create a custom theme. Based on provided inputs, generate a new theme similar to the ones above. Give the theme a similar name describing what the font/color combinations represent. Use any basic description provided to choose appropriate colors/fonts. After generating the theme, show it for review and verification. Following that, apply the theme as described above.
