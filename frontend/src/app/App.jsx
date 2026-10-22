import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppShell from "../components/layout/AppShell.jsx";
import DashboardPage from "../features/dashboard/DashboardPage.jsx";
import IdentificationPage from "../features/identification/IdentificationPage.jsx";
import ChatbotPage from "../features/chatbot/ChatbotPage.jsx";
import DiseasePage from "../features/disease/DiseasePage.jsx";
import QualityPage from "../features/quality/QualityPage.jsx";
import ForecastPage from "../features/forecast/ForecastPage.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<DashboardPage />} />
          <Route path="/c1/identify" element={<IdentificationPage />} />
          <Route path="/c1/chat" element={<ChatbotPage />} />
          <Route path="/c2/disease" element={<DiseasePage />} />
          <Route path="/c3/quality" element={<QualityPage />} />
          <Route path="/c4/forecast" element={<ForecastPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
