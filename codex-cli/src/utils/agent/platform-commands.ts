/**
 * Utility functions for handling platform-specific commands
 */

import { log } from "../logger/log.js";

/**
 * Map of Unix commands to their Windows equivalents
 */
const COMMAND_MAP: Record<string, string> = {
  ls: "dir",
  grep: "findstr",
  cat: "type",
  rm: "del",
  cp: "copy",
  mv: "move",
  touch: "echo.>",
  mkdir: "md",
};

/**
 * List of commands that are built into `cmd.exe` on Windows. These do not
 * correspond to standalone executables, meaning they must be executed through
 * the shell.
 */
const CMD_BUILTINS = new Set([
  "dir",
  "copy",
  "move",
  "del",
  "type",
  "echo.>",
  "md",
]);

/**
 * Map of common Unix command options to their Windows equivalents
 */
const OPTION_MAP: Record<string, Record<string, string>> = {
  ls: {
    "-l": "/p",
    "-a": "/a",
    "-R": "/s",
  },
  grep: {
    "-i": "/i",
    "-r": "/s",
  },
};

/**
 * Adapts a command for the current platform.
 * On Windows, this will translate Unix commands to their Windows equivalents.
 * On Unix-like systems, this will return the original command.
 *
 * @param command The command array to adapt
 * @returns The adapted command array
 */
export function adaptCommandForPlatform(command: Array<string>): Array<string> {
  // If not on Windows, return the original command
  if (process.platform !== "win32") {
    return command;
  }

  // Nothing to adapt if the command is empty
  if (command.length === 0) {
    return command;
  }

  const cmd = command[0];

  // If cmd is undefined or the command doesn't need adaptation, return it as is
  if (!cmd || !COMMAND_MAP[cmd]) {
    return command;
  }

  log(`Adapting command '${cmd}' for Windows platform`);

  // Create a new command array with the adapted command
  const adaptedCommand = [...command];
  adaptedCommand[0] = COMMAND_MAP[cmd];

  // Adapt options if needed
  const optionsForCmd = OPTION_MAP[cmd];
  if (optionsForCmd) {
    for (let i = 1; i < adaptedCommand.length; i++) {
      const option = adaptedCommand[i];
      if (option && optionsForCmd[option]) {
        adaptedCommand[i] = optionsForCmd[option];
      }
    }
  }

  log(`Adapted command: ${adaptedCommand.join(" ")}`);

  if (CMD_BUILTINS.has(adaptedCommand[0])) {
    // Wrap built-in commands so that `spawn` executes them via cmd.exe
    return ["cmd.exe", "/c", ...adaptedCommand];
  }

  return adaptedCommand;
}
