import { Header } from "./components/Header";
import { Toc } from "./components/Toc";
import { Footer } from "./components/Footer";
import { Problem } from "./sections/Problem";
import { Idea } from "./sections/Idea";
import { Reward } from "./sections/Reward";
import { Training } from "./sections/Training";
import { Setup } from "./sections/Setup";
import { H1Pareto } from "./sections/H1Pareto";
import { H2Compute } from "./sections/H2Compute";
import { H3ActionMix } from "./sections/H3ActionMix";
import { Conclusion } from "./sections/Conclusion";
import { Limitations } from "./sections/Limitations";

export default function App() {
  return (
    <>
      <Header />
      <Toc />
      <div className="mx-auto max-w-[780px] px-4 pb-16 pt-10 md:px-6">
        <Problem />
        <Idea />
        <Reward />
        <Training />
        <Setup />
        <H1Pareto />
        <H2Compute />
        <H3ActionMix />
        <Conclusion />
        <Limitations />
      </div>
      <Footer />
    </>
  );
}
