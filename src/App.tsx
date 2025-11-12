import { createBrowserRouter, RouterProvider } from "react-router-dom";
import HomePage from "./pages/HomePage";
import StationPage from "./pages/StationPage";
import NotFoundPage from "./pages/NotFoundPage";
import TripTypePage from "./pages/TripTypePage";
import DateTimePage from "./pages/DateTimePage";
import PassengerPage from "./pages/PassengerPage";
import SeatPage from "./pages/SeatPage";
import SeatListPage from "./pages/SeatListPage";
import DeparturePage from "./pages/DeparturePage";


const router = createBrowserRouter([
  { path: "/", element: <HomePage />, errorElement: <NotFoundPage /> },
  { path: "/departure", element: <DeparturePage /> },
  { path: "/triptype", element: <TripTypePage/>},
  { path: "/datetime", element: <DateTimePage/>},
  { path: "/passenger", element: <PassengerPage /> },
  { path: "/seat", element: <SeatPage/>},
  { path: "/seatlist", element: <SeatListPage/>},
  { path: "/"}

]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;
