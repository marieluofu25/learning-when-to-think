export const SECTION_NAV = [
  { id: "problem", label: "1. Problem" },
  { id: "idea", label: "2. Idea" },
  { id: "reward", label: "3. Reward" },
  { id: "train", label: "4. Training" },
  { id: "setup", label: "5. Setup" },
  { id: "h1", label: "6. H1" },
  { id: "h2", label: "7. H2" },
  { id: "h3", label: "8. H3" },
  { id: "conclusion", label: "9. Conclusion" },
  { id: "limits", label: "10. Limits" },
] as const;

export type SectionId = (typeof SECTION_NAV)[number]["id"];
