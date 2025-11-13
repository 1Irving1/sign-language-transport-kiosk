import HomeIcon from "./icons/HomeIcon";
import BackIcon from "./icons/backIcon";

interface HeaderProps {
  title: string;
}


export default function Header({ title }: HeaderProps) {
  return (
    <header className="bg-blue-50 w-full py-6 shadow-sm relative">
      <button className="absolute left-4 top-1/2 -translate-y-1/2">
        <BackIcon/>
      </button>
      
     
      <h1 className="text-center font-bold text-lg">{title}</h1>
      
     
      <button className="absolute right-4 top-1/2 -translate-y-1/2">
        <HomeIcon/>
      </button>
    </header>
  );
}