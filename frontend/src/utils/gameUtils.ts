export function parseGameSummary(summary: string) {
  const teamsPart = summary.split(' - ')[1];
  if (!teamsPart) return { away: '', home: '' };
  
  const cleanTeams = teamsPart.replace(/\s*\([^)]+\)\s*$/, '').trim();
  const [away, home] = cleanTeams.split(' @ ');
  return { 
    away: away?.replace(/\s*\(\d+\)/g, '').trim() || '', 
    home: home?.replace(/\s*\(\d+\)/g, '').trim() || '' 
  };
}

export function parseEspnName(espnName: string) {
  let [away, home] = espnName.split(' at ');
  // Handle Athletics edge case
  if (away === 'Athletics Athletics') away = 'Athletics';
  if (home === 'Athletics Athletics') home = 'Athletics';
  return { away: away?.trim() || '', home: home?.trim() || '' };
}

export function formatDateTime(dateTime: string) {
  try {
    return new Date(dateTime).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  } catch {
    return dateTime;
  }
}

export function getStatusColor(status: string) {
  const lowerStatus = status.toLowerCase();
  if (lowerStatus.includes('live') || lowerStatus.includes('in progress')) return 'live';
  if (lowerStatus.includes('final')) return 'final';
  return 'scheduled';
}

export function getPitchColor(result: string) {
  switch(result?.toLowerCase()) {
    case 'strike': return 'strike';
    case 'foul': return 'foul';
    case 'ball': return 'ball';
    case 'play': return 'play';
    default: return 'default';
  }
}